from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment2.modules.adversarial import search_fgsm_attack
from assignment2.modules.denoise import SpectralSubtractionDenoiser
from assignment2.modules.ipa_translation import CorpusTranslator, UnifiedIPAMapper
from assignment2.modules.lid_system import train_lid_model
from assignment2.modules.reporting import run_report_metrics
from assignment2.modules.speaker import SpeakerEmbeddingExtractor
from assignment2.modules.spoof import train_spoof_model
from assignment2.modules.stt_system import build_transcriber
from assignment2.modules.tts import GujaratiTTS
from assignment2.models.lid import FrameLevelLIDNet
from assignment2.utils.audio import load_audio, peak_normalize, save_audio


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def ensure_outputs_dir(config: dict) -> Path:
    out_dir = Path(config["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def command_train_lid(config: dict, device: str) -> None:
    audio_cfg = config["audio"]
    lid_cfg = config["lid"]
    train_lid_model(
        manifest_path=config["paths"]["lid_manifest"],
        checkpoint_path=config["paths"]["lid_checkpoint"],
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        win_length=audio_cfg["win_length"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        batch_size=lid_cfg["batch_size"],
        epochs=lid_cfg["epochs"],
        learning_rate=lid_cfg["learning_rate"],
        chunk_seconds=lid_cfg["chunk_seconds"],
        model_kwargs={
            "input_dim": lid_cfg["input_dim"],
            "conv_dim": lid_cfg["conv_dim"],
            "hidden_dim": lid_cfg["hidden_dim"],
            "num_languages": lid_cfg["num_languages"],
        },
        switch_loss_weight=lid_cfg["switch_loss_weight"],
        switch_boundary_radius_frames=lid_cfg.get("switch_boundary_radius_frames", 6),
        device=device,
    )
    print(f"Saved LID checkpoint to {config['paths']['lid_checkpoint']}")


def command_transcribe(config: dict, device: str) -> None:
    out_dir = ensure_outputs_dir(config)
    audio_cfg = config["audio"]
    decode_cfg = config["decoding"]
    syllabus_text = Path(config["paths"]["syllabus_text"]).read_text(encoding="utf-8")

    waveform, sample_rate = load_audio(
        config["paths"]["original_audio"],
        target_sr=audio_cfg["sample_rate"],
        mono=True,
    )
    noise_frames = max(1, int(audio_cfg["noise_estimate_sec"] * sample_rate / audio_cfg["hop_length"]))
    denoiser = SpectralSubtractionDenoiser(
        n_fft=audio_cfg["n_fft"],
        win_length=audio_cfg["win_length"],
        hop_length=audio_cfg["hop_length"],
        noise_estimate_frames=noise_frames,
    )
    denoised = peak_normalize(denoiser(waveform))
    denoised_path = out_dir / "original_segment_denoised.wav"
    save_audio(denoised_path, denoised, sample_rate)

    transcriber = build_transcriber(
        model_name=config["paths"]["stt_model_name"],
        syllabus_text=syllabus_text,
        ngram_order=decode_cfg.get("ngram_order", 3),
        beam_size=decode_cfg["beam_size"],
        lm_weight=decode_cfg["lm_weight"],
        length_penalty=decode_cfg["length_penalty"],
        term_bonus=decode_cfg["term_bonus"],
        device=device,
        backend=decode_cfg.get("backend", "auto"),
        language=decode_cfg.get("language"),
        task=decode_cfg.get("task", "transcribe"),
        initial_token_bias=decode_cfg.get("initial_token_bias", 0.25),
        max_new_tokens=decode_cfg.get("max_new_tokens", 256),
    )
    transcript = transcriber.transcribe_file(
        denoised_path,
        chunk_seconds=decode_cfg["chunk_seconds"],
        overlap_seconds=decode_cfg["chunk_overlap_seconds"],
    )
    transcript_path = Path(config["paths"]["transcript_output"])
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(transcript, encoding="utf-8")
    print(f"Saved transcript to {transcript_path}")


def command_task2(config: dict, text_file: str | None) -> None:
    transcript_path = Path(text_file or config["paths"]["transcript_output"])
    text = transcript_path.read_text(encoding="utf-8")
    mapper = UnifiedIPAMapper(config["paths"]["task2_corpus"])
    translator = CorpusTranslator(config["paths"]["task2_corpus"])
    ipa = mapper.text_to_ipa(text)
    gujarati = translator.translate(text)

    out_dir = ensure_outputs_dir(config)
    (out_dir / "transcript_unified_ipa.txt").write_text(ipa, encoding="utf-8")
    (out_dir / "translation_gujarati.txt").write_text(gujarati, encoding="utf-8")
    print(f"Saved IPA to {out_dir / 'transcript_unified_ipa.txt'}")
    print(f"Saved Gujarati translation to {out_dir / 'translation_gujarati.txt'}")


def command_speaker_embed(config: dict) -> None:
    extractor = SpeakerEmbeddingExtractor(
        checkpoint_path=config["paths"]["speaker_checkpoint"],
        sample_rate=config["audio"]["sample_rate"],
        embedding_dim=config["speaker"]["embedding_dim"],
    )
    embedding = extractor.extract(config["paths"]["student_voice_audio"])
    out_dir = ensure_outputs_dir(config)
    embedding_path = out_dir / "student_voice_embedding.pt"
    torch.save(embedding, embedding_path)
    print(f"Saved embedding to {embedding_path}")


def command_synthesize(config: dict, text_file: str | None) -> None:
    text_path = Path(text_file or config["paths"]["translation_output"])
    text = text_path.read_text(encoding="utf-8")
    tts = GujaratiTTS(
        model_name=config["paths"]["tts_model_name"],
        sample_rate=config["tts"]["sample_rate"],
        speaker_id=config["tts"]["speaker_id"],
    )
    out_dir = ensure_outputs_dir(config)
    output_path = out_dir / "output_LRL_cloned.wav"
    flat_output_path = out_dir / "output_LRL_flat.wav"
    tts.synthesize_with_prosody(
        text=text,
        reference_professor_audio=config["paths"]["original_audio"],
        output_path=output_path,
        flat_output_path=flat_output_path,
        speaking_rate=config["tts"]["speaking_rate"],
        hop_ms=config["audio"]["hop_ms"],
        frame_length=config["audio"]["win_length"],
        hop_length=config["audio"]["hop_length"],
    )
    print(f"Saved synthesized audio to {output_path}")


def command_train_spoof(config: dict, device: str) -> None:
    checkpoint_path = Path(config["paths"].get("spoof_checkpoint", ROOT / "models" / "anti_spoof_lfcc.pt"))
    train_spoof_model(
        manifest_path=config["paths"]["spoof_manifest"],
        checkpoint_path=checkpoint_path,
        sample_rate=config["audio"]["sample_rate"],
        n_lfcc=config["spoof"]["n_lfcc"],
        batch_size=config["spoof"]["batch_size"],
        epochs=config["spoof"]["epochs"],
        learning_rate=config["spoof"]["learning_rate"],
        device=device,
    )
    print(f"Saved anti-spoof checkpoint to {checkpoint_path}")


def command_attack(config: dict, device: str) -> None:
    lid_cfg = config["lid"]
    audio_cfg = config["audio"]
    model = FrameLevelLIDNet(
        input_dim=lid_cfg["input_dim"],
        conv_dim=lid_cfg["conv_dim"],
        hidden_dim=lid_cfg["hidden_dim"],
        num_languages=lid_cfg["num_languages"],
    ).to(device)
    checkpoint = Path(config["paths"]["lid_checkpoint"])
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    waveform, sample_rate = load_audio(
        config["paths"]["original_audio"],
        target_sr=audio_cfg["sample_rate"],
        mono=True,
    )
    search_result = search_fgsm_attack(
        model=model,
        waveform=waveform.to(device),
        sample_rate=sample_rate,
        source_label=config["adversarial"].get("source_label", 0),
        target_label=config["adversarial"]["target_label"],
        epsilon_grid=config["adversarial"]["epsilon_grid"],
        snr_threshold_db=config["adversarial"]["snr_threshold_db"],
        n_fft=audio_cfg["n_fft"],
        win_length=audio_cfg["win_length"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        segment_seconds=config["adversarial"].get("segment_seconds", 5.0),
        stride_seconds=config["adversarial"].get("stride_seconds", 2.5),
        max_candidate_segments=config["adversarial"].get("max_candidate_segments", 8),
        attack_steps=config["adversarial"].get("attack_steps", 10),
        success_ratio_threshold=config["adversarial"].get("success_ratio_threshold", 0.5),
    )
    if search_result is not None and search_result.successful_waveform is not None and search_result.successful_epsilon is not None:
        out_path = ensure_outputs_dir(config) / "lid_fgsm_attack.wav"
        save_audio(out_path, search_result.successful_waveform.cpu(), sample_rate)
        print(
            "Attack succeeded at "
            f"epsilon={search_result.successful_epsilon} on "
            f"{search_result.segment_start_sec:.2f}-{search_result.segment_end_sec:.2f}s; saved {out_path}"
        )
    else:
        print("No epsilon satisfied both flip and SNR constraints.")


def command_evaluate(config: dict, device: str, reference_text_file: str | None, hypothesis_text_file: str | None) -> None:
    metrics = run_report_metrics(
        config=config,
        device=device,
        reference_text_path=reference_text_file,
        hypothesis_text_path=hypothesis_text_file,
    )
    print(json.dumps(metrics, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech Understanding Assignment 2 pipeline")
    parser.add_argument(
        "command",
        choices=["train_lid", "transcribe", "task2", "speaker_embed", "synthesize", "train_spoof", "attack", "evaluate"],
    )
    parser.add_argument("--config", default="configs/assignment2_config.json")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--reference-text-file", default=None)
    parser.add_argument("--hypothesis-text-file", default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    if args.command == "train_lid":
        command_train_lid(config, device)
    elif args.command == "transcribe":
        command_transcribe(config, device)
    elif args.command == "task2":
        command_task2(config, args.text_file)
    elif args.command == "speaker_embed":
        command_speaker_embed(config)
    elif args.command == "synthesize":
        command_synthesize(config, args.text_file)
    elif args.command == "train_spoof":
        command_train_spoof(config, device)
    elif args.command == "attack":
        command_attack(config, device)
    elif args.command == "evaluate":
        command_evaluate(config, device, args.reference_text_file, args.hypothesis_text_file)


if __name__ == "__main__":
    main()
