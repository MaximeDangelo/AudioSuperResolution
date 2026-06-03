#!/usr/bin/env python3
"""
Fine-tuning MetricGAN+ pour denoising radio ATC/militaire.

Charge les poids pre-entraines du modele MetricGAN+ (Voicebank),
puis fine-tune sur le dataset radio synthetique.

Usage:
    python train_metricgan.py hparams/train_radio.yaml

Auteurs: adapte de recipes/Voicebank/enhance/MetricGAN/train.py
"""

import os
import pickle
import shutil
import sys
from enum import Enum, auto

import torch
from hyperpyyaml import load_hyperpyyaml
from pesq import pesq

import speechbrain as sb
from speechbrain.dataio import audio_io
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.processing.features import spectral_magnitude
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import MetricStats


def pesq_eval(pred_wav, target_wav):
    """Normalized PESQ (to 0-1)"""
    return (
        pesq(fs=16000, ref=target_wav.numpy(), deg=pred_wav.numpy(), mode="wb")
        + 0.5
    ) / 5


class SubStage(Enum):
    """For keeping track of training stage progress"""
    GENERATOR = auto()
    CURRENT = auto()
    HISTORICAL = auto()


class MetricGanBrain(sb.Brain):
    def load_history(self):
        if os.path.isfile(self.hparams.historical_file):
            with open(self.hparams.historical_file, "rb") as fp:
                self.historical_set = pickle.load(fp)

    def compute_feats(self, wavs):
        """Feature computation pipeline"""
        feats = self.hparams.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        return feats

    def compute_forward(self, batch, stage):
        "Given an input batch computes the enhanced signal"
        batch = batch.to(self.device)

        if self.sub_stage == SubStage.HISTORICAL:
            predict_wav, lens = batch.enh_sig
        else:
            noisy_wav, lens = batch.noisy_sig
            noisy_spec = self.compute_feats(noisy_wav)

            mask = self.modules.generator(noisy_spec, lengths=lens)
            mask = mask.clamp(min=self.hparams.min_mask).squeeze(2)
            predict_spec = torch.mul(mask, noisy_spec)

            predict_wav = self.hparams.resynth(
                torch.expm1(predict_spec), noisy_wav
            )

        return predict_wav

    def compute_objectives(self, predictions, batch, stage, optim_name=""):
        "Given the network predictions and targets compute the total loss"
        predict_wav = predictions
        predict_spec = self.compute_feats(predict_wav)

        clean_wav, lens = batch.clean_sig
        clean_spec = self.compute_feats(clean_wav)
        clean_paths = batch.clean_wav

        ids = self.compute_ids(batch.id, optim_name)

        if optim_name == "generator":
            target_score = torch.ones(self.batch_size, 1, device=self.device)
            est_score = self.est_score(predict_spec, clean_spec)
            self.mse_metric.append(
                ids, predict_spec, clean_spec, lens, reduction="batch"
            )
            mse_cost = self.hparams.compute_cost(predict_spec, clean_spec, lens)

        elif optim_name == "D_clean":
            target_score = torch.ones(self.batch_size, 1, device=self.device)
            est_score = self.est_score(clean_spec, clean_spec)

        elif optim_name == "D_enh" and self.sub_stage == SubStage.CURRENT:
            target_score = self.score(ids, predict_wav, clean_wav, lens)
            est_score = self.est_score(predict_spec, clean_spec)
            self.write_wavs(ids, predict_wav, clean_paths, target_score, lens)

        elif optim_name == "D_enh" and self.sub_stage == SubStage.HISTORICAL:
            target_score = batch.score.unsqueeze(1).float()
            est_score = self.est_score(predict_spec, clean_spec)

        elif optim_name == "D_noisy":
            noisy_wav, _ = batch.noisy_sig
            noisy_spec = self.compute_feats(noisy_wav)
            target_score = self.score(ids, noisy_wav, clean_wav, lens)
            est_score = self.est_score(noisy_spec, clean_spec)
            self.save_noisy_scores(ids, target_score)

        if stage == sb.Stage.TRAIN:
            cost = self.hparams.compute_cost(est_score, target_score)
            if optim_name == "generator":
                cost += self.hparams.mse_weight * mse_cost
                self.metrics["G"].append(cost.detach())
            else:
                self.metrics["D"].append(cost.detach())

        if stage != sb.Stage.TRAIN:
            cost = self.hparams.compute_si_snr(predict_wav, clean_wav, lens)
            self.stoi_metric.append(
                batch.id, predict_wav, clean_wav, lens, reduction="batch"
            )
            self.pesq_metric.append(
                batch.id, predict=predict_wav, target=clean_wav, lengths=lens
            )

            lens = lens * clean_wav.shape[1]
            for name, pred_wav, length in zip(batch.id, predict_wav, lens):
                name += ".wav"
                enhance_path = os.path.join(self.hparams.enhanced_folder, name)
                audio_io.save(
                    enhance_path,
                    torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                    16000,
                )

        return cost

    def compute_ids(self, batch_id, optim_name):
        if optim_name == "D_enh":
            return [f"{uid}@{self.epoch}" for uid in batch_id]
        return batch_id

    def save_noisy_scores(self, batch_id, scores):
        for i, score in zip(batch_id, scores):
            self.noisy_scores[i] = score

    def score(self, batch_id, deg_wav, ref_wav, lens):
        new_ids = [
            i
            for i, d in enumerate(batch_id)
            if d not in self.historical_set and d not in self.noisy_scores
        ]

        if len(new_ids) == 0:
            pass
        elif self.hparams.target_metric == "pesq":
            self.target_metric.append(
                ids=[batch_id[i] for i in new_ids],
                predict=deg_wav[new_ids].detach(),
                target=ref_wav[new_ids].detach(),
                lengths=lens[new_ids],
            )
            score = torch.tensor(
                [[s] for s in self.target_metric.scores], device=self.device
            )
        elif self.hparams.target_metric == "stoi":
            self.target_metric.append(
                [batch_id[i] for i in new_ids],
                deg_wav[new_ids],
                ref_wav[new_ids],
                lens[new_ids],
                reduction="batch",
            )
            score = torch.tensor(
                [[-s] for s in self.target_metric.scores],
                device=self.device,
            )
        else:
            raise ValueError("Expected 'pesq' or 'stoi' for target_metric")

        self.target_metric.clear()

        final_score = []
        for i, d in enumerate(batch_id):
            if d in self.historical_set:
                final_score.append([self.historical_set[d]["score"]])
            elif d in self.noisy_scores:
                final_score.append([self.noisy_scores[d]])
            else:
                final_score.append([score[new_ids.index(i)]])

        return torch.tensor(final_score, device=self.device)

    def est_score(self, deg_spec, ref_spec):
        combined_spec = torch.cat(
            [deg_spec.unsqueeze(1), ref_spec.unsqueeze(1)], 1
        )
        return self.modules.discriminator(combined_spec)

    def write_wavs(self, batch_id, wavs, clean_paths, scores, lens):
        lens = lens * wavs.shape[1]
        record = {}
        for i, (name, pred_wav, clean_path, length) in enumerate(
            zip(batch_id, wavs, clean_paths, lens)
        ):
            path = os.path.join(self.hparams.MetricGAN_folder, name + ".wav")
            data = torch.unsqueeze(pred_wav[: int(length)].cpu(), 0)
            audio_io.save(path, data.detach(), self.hparams.Sample_rate)
            score = float(scores[i][0])
            record[name] = {
                "enh_wav": path,
                "score": score,
                "clean_wav": clean_path,
            }
        self.historical_set.update(record)
        with open(self.hparams.historical_file, "wb") as fp:
            pickle.dump(self.historical_set, fp)

    def fit_batch(self, batch):
        "Compute gradients and update either D or G based on sub-stage."
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss_tracker = 0
        if self.sub_stage == SubStage.CURRENT:
            for mode in ["clean", "enh", "noisy"]:
                loss = self.compute_objectives(
                    predictions, batch, sb.Stage.TRAIN, f"D_{mode}"
                )
                self.d_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.max_grad_norm
                )
                self.d_optimizer.step()
                loss_tracker += loss.detach() / 3
        elif self.sub_stage == SubStage.HISTORICAL:
            loss = self.compute_objectives(
                predictions, batch, sb.Stage.TRAIN, "D_enh"
            )
            self.d_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.modules.parameters(), self.max_grad_norm
            )
            self.d_optimizer.step()
            loss_tracker += loss.detach()
        elif self.sub_stage == SubStage.GENERATOR:
            for name, param in self.modules.generator.named_parameters():
                if "Learnable_sigmoid" in name:
                    param.data = torch.clamp(param, max=3.5)
                    param.data[param != param] = 3.5

            loss = self.compute_objectives(
                predictions, batch, sb.Stage.TRAIN, "generator"
            )
            self.g_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.modules.parameters(), self.max_grad_norm
            )
            self.g_optimizer.step()
            loss_tracker += loss.detach()

        return loss_tracker

    def on_stage_start(self, stage, epoch=None):
        self.mse_metric = MetricStats(metric=self.hparams.compute_cost)
        self.metrics = {"G": [], "D": []}

        if stage == sb.Stage.TRAIN:
            if self.hparams.target_metric == "pesq":
                self.target_metric = MetricStats(
                    metric=pesq_eval, n_jobs=hparams["n_jobs"], batch_eval=False
                )
            elif self.hparams.target_metric == "stoi":
                self.target_metric = MetricStats(metric=stoi_loss)

            if self.sub_stage == SubStage.GENERATOR:
                self.epoch = epoch
                self.train_discriminator()
                self.sub_stage = SubStage.GENERATOR
                print("Generator training by current data...")

        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(
                metric=pesq_eval, n_jobs=hparams["n_jobs"], batch_eval=False
            )
            self.stoi_metric = MetricStats(metric=stoi_loss)

    def train_discriminator(self):
        print("Discriminator training by current data...")
        self.sub_stage = SubStage.CURRENT
        self.fit(
            range(1),
            self.train_set,
            train_loader_kwargs=self.hparams.dataloader_options,
        )

        if self.historical_set:
            print("Discriminator training by historical data...")
            self.sub_stage = SubStage.HISTORICAL
            self.fit(
                range(1),
                self.historical_set,
                train_loader_kwargs=self.hparams.dataloader_options,
            )

        print("Discriminator training by current data again...")
        self.sub_stage = SubStage.CURRENT
        self.fit(
            range(1),
            self.train_set,
            train_loader_kwargs=self.hparams.dataloader_options,
        )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if self.sub_stage != SubStage.GENERATOR:
            return

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            g_loss = torch.tensor(self.metrics["G"])
            d_loss = torch.tensor(self.metrics["D"])
            print("Avg G loss: %.3f" % torch.mean(g_loss))
            print("Avg D loss: %.3f" % torch.mean(d_loss))
        else:
            stats = {
                "SI-SNR": -stage_loss,
                "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
                "stoi": -self.stoi_metric.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            if self.hparams.use_tensorboard:
                valid_stats = {
                    "SI-SNR": -stage_loss,
                    "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
                    "stoi": -self.stoi_metric.summarize("average"),
                }
                self.hparams.tensorboard_train_logger.log_stats(valid_stats)
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stats, max_keys=[self.hparams.target_metric]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        if stage == sb.Stage.TRAIN:
            if self.sub_stage == SubStage.HISTORICAL:
                dataset = sb.dataio.dataset.DynamicItemDataset(
                    data=dataset,
                    dynamic_items=[enh_pipeline],
                    output_keys=[
                        "id",
                        "enh_sig",
                        "clean_sig",
                        "score",
                        "clean_wav",
                    ],
                )
                samples = round(len(dataset) * self.hparams.history_portion)
                samples = max(samples, 1)
            else:
                samples = self.hparams.number_of_samples

            epoch = self.hparams.epoch_counter.current
            weights = torch.ones(len(dataset))
            replacement = samples > len(dataset)
            sampler = ReproducibleWeightedRandomSampler(
                weights,
                epoch=epoch,
                replacement=replacement,
                num_samples=samples,
            )
            loader_kwargs["sampler"] = sampler

            if self.sub_stage == SubStage.GENERATOR:
                self.train_sampler = sampler

        return super().make_dataloader(
            dataset, stage, ckpt_prefix, **loader_kwargs
        )

    def on_fit_start(self):
        if self.sub_stage == SubStage.GENERATOR:
            super().on_fit_start()

    def init_optimizers(self):
        self.g_optimizer = self.hparams.g_opt_class(
            self.modules.generator.parameters()
        )
        self.d_optimizer = self.hparams.d_opt_class(
            self.modules.discriminator.parameters()
        )
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("g_opt", self.g_optimizer)
            self.checkpointer.add_recoverable("d_opt", self.d_optimizer)

    def zero_grad(self, set_to_none=False):
        self.g_optimizer.zero_grad(set_to_none)
        self.d_optimizer.zero_grad(set_to_none)


# Audio pipelines
@sb.utils.data_pipeline.takes("noisy_wav", "clean_wav")
@sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
def audio_pipeline(noisy_wav, clean_wav):
    yield sb.dataio.dataio.read_audio(noisy_wav)
    yield sb.dataio.dataio.read_audio(clean_wav)


@sb.utils.data_pipeline.takes("enh_wav", "clean_wav")
@sb.utils.data_pipeline.provides("enh_sig", "clean_sig")
def enh_pipeline(enh_wav, clean_wav):
    yield sb.dataio.dataio.read_audio(enh_wav)
    yield sb.dataio.dataio.read_audio(clean_wav)


def dataio_prep(hparams):
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[audio_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig", "clean_wav"],
        )
    return datasets


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def load_pretrained_weights(brain, pretrained_source="speechbrain/metricgan-plus-voicebank"):
    """Charge les poids pre-entraines MetricGAN+ depuis HuggingFace."""
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    print(f"\nChargement des poids pre-entraines depuis {pretrained_source}...")
    pretrained = SpectralMaskEnhancement.from_hparams(
        source=pretrained_source,
        savedir="pretrained_metricgan",
    )

    # Copier les poids du generateur (mods.enhance_model -> modules.generator)
    brain.modules.generator.load_state_dict(
        pretrained.mods.enhance_model.state_dict(),
        strict=False,
    )
    print("  Poids du generateur charges avec succes.")
    del pretrained


# Recipe begins!
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)

    # Prepare data (generate JSON annotations)
    from prepare_data import main as prepare_main
    run_on_main(prepare_main)

    datasets = dataio_prep(hparams)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger
        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    run_on_main(create_folder, kwargs={"folder": hparams["enhanced_folder"]})

    se_brain = MetricGanBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Charger les poids pre-entraines MetricGAN+ (fine-tuning)
    load_pretrained_weights(se_brain)

    se_brain.train_set = datasets["train"]
    se_brain.historical_set = {}
    se_brain.noisy_scores = {}
    se_brain.batch_size = hparams["dataloader_options"]["batch_size"]
    se_brain.sub_stage = SubStage.GENERATOR

    if not os.path.isfile(hparams["historical_file"]):
        if os.path.isdir(hparams["MetricGAN_folder"]):
            shutil.rmtree(hparams["MetricGAN_folder"])
    run_on_main(create_folder, kwargs={"folder": hparams["MetricGAN_folder"]})

    se_brain.load_history()

    # Fine-tuning
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    # Evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key=hparams["target_metric"],
        test_loader_kwargs=hparams["dataloader_options"],
    )
