"""Cruzamento de frames e anotações GSR v1.3 e cinemática descritiva por tracklet."""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
WINDOW = 11
SPEED_QC = 12.0
ACCEL_QC = 12.0


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_points(obj):
    info = obj["info"]
    if float(info["version"]) < 1.3:
        raise ValueError("Usar anotações com consistência temporal revisada, versão >= 1.3.")
    fps = float(info["frame_rate"])
    if not 0 < fps <= 120:
        raise ValueError("FPS inválido.")
    images = pd.DataFrame([{"image_id": str(i["image_id"]),
                            "frame": int(Path(i["file_name"]).stem)} for i in obj["images"]])
    assert images.image_id.is_unique and images.frame.is_unique
    records = []
    missing_coordinates = 0
    for a in obj["annotations"]:
        attrs = a.get("attributes") or {}
        if attrs.get("role") not in {"player", "goalkeeper"} or a.get("track_id") is None:
            continue
        bbox = a.get("bbox_pitch") or {}
        x, y = bbox.get("x_bottom_middle"), bbox.get("y_bottom_middle")
        if x is None or y is None or not np.isfinite([x, y]).all():
            missing_coordinates += 1
            continue
        records.append({"image_id": str(a["image_id"]), "track_id": int(a["track_id"]),
                        "role": attrs["role"], "team": attrs.get("team") or "unknown",
                        "jersey": attrs.get("jersey"), "x_m": float(x), "y_m": float(y)})
    points = pd.DataFrame(records)
    if points.empty:
        return points, fps, missing_coordinates
    points = points.merge(images, on="image_id", how="left", validate="many_to_one", indicator=True)
    assert points["_merge"].eq("both").all(), "Anotação sem frame correspondente."
    assert not points.duplicated(["track_id", "frame"]).any(), "Identidade duplicada no mesmo frame."
    points = points.drop(columns="_merge").sort_values(["track_id", "frame"])
    # Margem geométrica explícita para jogadores próximos às linhas laterais/fundo.
    points["coordinate_qc"] = points.x_m.abs().le(60) & points.y_m.abs().le(45)
    return points, fps, missing_coordinates


def segment_motion(track, fps):
    """Não conecta gaps, mudanças de equipe/papel ou pontos fora da margem geométrica."""
    track = track.sort_values("frame").copy()
    valid = track[track.coordinate_qc].copy()
    if valid.empty:
        return pd.DataFrame(), 0
    starts = valid.frame.diff().ne(1) | valid.team.ne(valid.team.shift()) | valid.role.ne(valid.role.shift())
    valid["segment"] = starts.cumsum()
    outputs = []
    segments = 0
    for _, seg in valid.groupby("segment", sort=False):
        if len(seg) < 2 * WINDOW:
            continue
        segments += 1
        x, y = seg.x_m.to_numpy(), seg.y_m.to_numpy()
        # Derivadas centrais suavizadas: análise retrospectiva, não filtro online.
        vx = savgol_filter(x, WINDOW, 2, deriv=1, delta=1/fps)
        vy = savgol_filter(y, WINDOW, 2, deriv=1, delta=1/fps)
        speed = np.hypot(vx, vy)
        accel = savgol_filter(speed, WINDOW, 2, deriv=1, delta=1/fps)
        raw_speed = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y)) * fps]
        # Duas operações locais de janela 11: retirar 10 frames em cada borda.
        trim = WINDOW - 1
        core = seg.iloc[trim:-trim].copy()
        core["speed_m_s"] = speed[trim:-trim]
        core["tangential_accel_m_s2"] = accel[trim:-trim]
        core["native_difference_speed_m_s"] = raw_speed[trim:-trim]
        core["motion_qc"] = core.speed_m_s.le(SPEED_QC) & core.tangential_accel_m_s2.abs().le(ACCEL_QC)
        outputs.append(core)
    return (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()), segments


def analyze_clip(obj, split):
    points, fps, missing = load_points(obj)
    info = obj["info"]
    clip = info["name"]
    counts = {"split": split, "clip": clip, "game_id": str(info.get("game_id")),
              "frames": len(obj["images"]), "fps": fps, "version": info["version"],
              "player_annotations_with_coordinates": len(points), "missing_pitch_coordinates": missing,
              "outside_coordinate_margin": int((~points.coordinate_qc).sum()) if len(points) else 0,
              "motion_points_before_qc": 0, "motion_points_after_qc": 0}
    rows, samples = [], []
    histogram = np.zeros((60, 80), dtype=np.int64)
    if points.empty:
        return rows, counts, histogram, pd.DataFrame()
    for track_id, track in points.groupby("track_id", sort=True):
        motion, nsegments = segment_motion(track, fps)
        row = {"split": split, "clip": clip, "track_id": int(track_id),
               "role": str(track.role.mode().iloc[0]), "team": str(track.team.mode().iloc[0]),
               "frames_observed": len(track), "frame_coverage_fraction": len(track) / len(obj["images"]),
               "segments_analyzed": nsegments, "motion_points": len(motion),
               "usable_motion_points": 0, "observed_distance_m": 0.0,
               "connected_observation_s": 0.0,
               "mean_speed_m_s": None, "p95_speed_m_s": None,
               "p95_abs_tangential_accel_m_s2": None,
               "native_difference_p95_speed_m_s": None}
        if len(motion):
            good = motion[motion.motion_qc].copy()
            counts["motion_points_before_qc"] += len(motion)
            counts["motion_points_after_qc"] += len(good)
            row["usable_motion_points"] = len(good)
            if len(good):
                row.update(mean_speed_m_s=float(good.speed_m_s.mean()), p95_speed_m_s=float(good.speed_m_s.quantile(.95)),
                           p95_abs_tangential_accel_m_s2=float(good.tangential_accel_m_s2.abs().quantile(.95)),
                           native_difference_p95_speed_m_s=float(good.native_difference_speed_m_s.quantile(.95)))
                adjacent = good.frame.diff().eq(1) & good.segment.eq(good.segment.shift())
                # Integral por trapézios somente entre amostras adjacentes aceitas.
                ds = (good.speed_m_s + good.speed_m_s.shift()) / (2 * fps)
                row["observed_distance_m"] = float(ds[adjacent].sum())
                row["connected_observation_s"] = float(adjacent.sum() / fps)
                histogram += np.histogram2d(good.speed_m_s, good.tangential_accel_m_s2,
                                           bins=[np.linspace(0,12,61), np.linspace(-12,12,81)])[0].astype(np.int64)
                if len(samples) < 3:
                    sample = good.iloc[::max(1, len(good)//100)].copy()
                    sample["clip"], sample["split"], sample["time_s"] = clip, split, sample.frame / fps
                    samples.append(sample)
        rows.append(row)
    return rows, counts, histogram, pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()


def plots(tracks, clips, histogram, sample):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(tracks.frame_coverage_fraction, bins=np.linspace(0,1,21), color="#237b87", edgecolor="white")
    axes[0].set(xlabel="Fração de frames em que a trajetória é observada", ylabel="Trajetórias locais",
                title="Cobertura visual; não são atletas únicos")
    image = axes[1].imshow(np.log1p(histogram.T), origin="lower", aspect="auto", extent=[0,12,-12,12], cmap="viridis")
    axes[1].set(xlabel="Velocidade suavizada (m/s)", ylabel="Aceleração tangencial (m/s²)",
                title="Densidade descritiva de movimento (log de contagem)")
    fig.colorbar(image, ax=axes[1], shrink=.8)
    fig.tight_layout(); fig.savefig(ROOT / "figures/cobertura_e_movimento.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if len(sample):
        for color_index, (tid, t) in enumerate(sample.groupby("track_id")):
            # IDs só são únicos dentro de um clip; sample contém exatamente um clip.
            color = plt.get_cmap("tab10")(color_index)
            for j, (_, segment) in enumerate(t.groupby("segment")):
                label = f"trajetória {tid}" if j == 0 else None
                # Pontos sem linhas: a amostragem visual nunca conecta gaps/QC.
                axes[0].plot(segment.x_m, segment.y_m, ".", ms=3, color=color, label=label)
                axes[1].plot(segment.time_s, segment.speed_m_s, ".", ms=3, color=color, label=label)
        axes[0].set_title(f"Posições anotadas: {sample['clip'].iloc[0]}")
    axes[0].set(xlim=(-60,60), ylim=(-45,45), xlabel="x no campo (m)", ylabel="y no campo (m)")
    axes[0].add_patch(plt.Rectangle((-52.5,-34),105,68,fill=False,color="gray"))
    axes[1].set(xlabel="Tempo local do clip (s)", ylabel="Velocidade suavizada (m/s)", title="Trechos observados; sem preencher gaps")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(ROOT / "figures/exemplo_trajetorias.png", dpi=220); plt.close(fig)


def run():
    for name in ["results", "figures", "data/processed"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "results/download_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("download_complete"):
        raise ValueError("Concluir download_annotations.py antes da análise.")
    all_rows, all_counts, sample = [], [], pd.DataFrame()
    histogram = np.zeros((60,80), dtype=np.int64)
    for i, item in enumerate(manifest["files"], 1):
        source = ROOT / "data/raw" / item["split"] / f"{item['clip']}.json.gz"
        with gzip.open(source, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != item["sha256_json"]:
            raise ValueError(f"Hash divergente: {source}")
        obj = json.loads(data)
        rows, counts, hist, example = analyze_clip(obj, item["split"])
        all_rows.extend(rows); all_counts.append(counts); histogram += hist
        if sample.empty and len(example):
            sample = example
        print(f"{i}/{len(manifest['files'])}: {item['split']}/{item['clip']}, {len(rows)} trajetórias", flush=True)
    tracks, clips = pd.DataFrame(all_rows), pd.DataFrame(all_counts)
    assert not tracks.duplicated(["split", "clip", "track_id"]).any()
    assert not clips.duplicated(["split", "clip"]).any()
    assert tracks.frame_coverage_fraction.between(0,1).all()
    assert tracks.usable_motion_points.sum() == int(histogram.sum())
    good = tracks[tracks.usable_motion_points.gt(0)]
    summary = {"dataset": manifest["repo"], "revision": manifest["revision"], "analysis_date": "2026-11-12",
               "source_scope": manifest["scope"], "annotation_versions": sorted(clips.version.unique().tolist()),
               "clips_by_split": clips.groupby("split").size().to_dict(), "clips": len(clips),
               "frames": int(clips.frames.sum()), "games_observed": int(clips.game_id.nunique()),
               "local_tracklets": len(tracks), "tracklets_with_motion": len(good),
               "player_annotations_with_coordinates": int(clips.player_annotations_with_coordinates.sum()),
               "missing_pitch_coordinates": int(clips.missing_pitch_coordinates.sum()),
               "outside_coordinate_margin": int(clips.outside_coordinate_margin.sum()),
               "motion_points_before_qc": int(clips.motion_points_before_qc.sum()),
               "motion_points_after_qc": int(clips.motion_points_after_qc.sum()),
               "median_frame_coverage": float(tracks.frame_coverage_fraction.median()),
               "median_track_mean_speed_m_s": float(good.mean_speed_m_s.median()),
               "median_track_p95_speed_m_s": float(good.p95_speed_m_s.median()),
               "connected_observation_s_sum": float(tracks.connected_observation_s.sum()),
               "kinematics": {"coordinate_source": "annotated bbox_pitch bottom_middle, v1.3",
                              "join": "annotations.image_id -> images.image_id, many_to_one",
                              "identity_key": ["split","clip","track_id"], "window_frames": WINDOW,
                              "polynomial_order": 2, "trim_each_edge_frames": WINDOW-1,
                              "segment_min_frames": 2*WINDOW, "max_speed_qc_m_s": SPEED_QC,
                              "max_abs_acceleration_qc_m_s2": ACCEL_QC,
                              "coordinate_margin_m": {"abs_x_max":60,"abs_y_max":45}},
               "injury_target_available": False, "fatigue_target_available": False,
               "clinical_7d_eligible": False, "tracking_model_run": False,
               "limitations": ["Annotated positions, not predictions from an automatic tracker or GPS",
                               "30-second clips do not establish within-match fatigue evolution",
                               "Track IDs and jersey/team do not establish cross-clip clinical identity",
                               "Filtering uses future neighbors and is descriptive offline only",
                               "No external kinematic reference to validate derived speed/acceleration",
                               "Image frames and challenge split not downloaded"]}
    save(ROOT / "results/summary.json", summary)
    save(ROOT / "results/environment.json", {n:importlib.metadata.version(n) for n in ["numpy","pandas","scipy","matplotlib","requests"]})
    tracks.to_csv(ROOT / "data/processed/tracklet_features.csv", index=False)
    clips.to_csv(ROOT / "data/processed/clip_audit.csv", index=False)
    sample.to_csv(ROOT / "data/processed/example_trajectories.csv", index=False)
    np.save(ROOT / "data/processed/speed_acceleration_histogram.npy", histogram)
    plots(tracks, clips, histogram, sample)
    lines = ["# Resultados executados - SoccerNet-GSR", "", "Cruzamento de anotações de referência e frames; sem modelo de lesão/fadiga.", "",
             f"- Clips: {len(clips)}; distribuição: {summary['clips_by_split']}.",
             f"- Frames declarados: {summary['frames']:,}; jogos representados: {summary['games_observed']}.",
             f"- Trajetórias locais: {len(tracks):,}; com movimento utilizável: {len(good):,}.",
             f"- Anotações de jogador/goleiro com coordenadas: {summary['player_annotations_with_coordinates']:,}.",
             f"- Anotações sem coordenadas no campo: {summary['missing_pitch_coordinates']:,}.",
             f"- Pontos derivados antes/depois do controle cinemático: {summary['motion_points_before_qc']:,} / {summary['motion_points_after_qc']:,}.",
             f"- Cobertura mediana por trajetória: {summary['median_frame_coverage']:.1%} dos frames do clip.",
             f"- Mediana das velocidades médias por trajetória com movimento: {summary['median_track_mean_speed_m_s']:.3f} m/s.",
             f"- Mediana dos percentis 95 de velocidade por trajetória: {summary['median_track_p95_speed_m_s']:.3f} m/s.", "",
             "Não são atletas únicos, médias de uma partida completa, estimativas validadas por GPS ou escores de fadiga/lesão."]
    (ROOT / "results/RESULTADOS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    run()
