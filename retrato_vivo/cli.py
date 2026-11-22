"""Linha de comando: testa o pipeline sem precisar da GUI.

Exemplos::

    python -m retrato_vivo restaurar foto.jpg -o saida --painel --explicar
    python -m retrato_vivo restaurar pasta_de_fotos -o saida --sensibilidade 0.7
    python -m retrato_vivo restaurar foto_pb.jpg -o saida --colorir --intensidade-cor 0.7
    python -m retrato_vivo restaurar foto_pequena.jpg -o saida --ampliar --modelo-sr edsr
    python -m retrato_vivo restaurar foto.jpg -o saida --bordas   # Canny x HED, antes e depois
    python -m retrato_vivo restaurar foto_do_celular.jpg -o saida --perspectiva
    python -m retrato_vivo avaliar foto_boa.jpg -o avaliacao
    python -m retrato_vivo envelhecer foto_boa.jpg -o antiga.png
    python -m retrato_vivo vivo foto.jpg -o saida                 # Retrato Vivo (vídeo 3D)
    python -m retrato_vivo viajar selfie.jpg --epoca 1920 -o 1920.png   # Máquina do tempo
    python -m retrato_vivo modelos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import metrics
from .aging import AgingConfig, age_photo
from .io_utils import list_images, load_image, resize_max_side, save_image
from .pipeline import RestorationConfig, RestorationResult, restore
from .visualize import side_by_side, steps_panel


def _config_from_args(args: argparse.Namespace) -> RestorationConfig:
    cfg = RestorationConfig(max_side=args.max_lado, inpaint_method=args.metodo)
    cfg.damage.sensitivity = args.sensibilidade
    cfg.damage.detect_large_losses = args.perdas_grandes
    cfg.correct_perspective = getattr(args, "perspectiva", False)    # só o "restaurar" tem a opção
    return cfg


def explain(res: RestorationResult) -> str:
    """Texto que conta o que foi medido na foto e o que cada etapa decidiu."""
    d = res.diagnosis
    lines = [
        "  diagnóstico:",
        f"    tipo            {'preto e branco / sépia' if d.monochrome else 'colorida'}"
        f" (variação de cor {d.chroma_std:.1f}, {d.off_tone_pct:.1f}% fora do tom dominante)",
        f"    dominante de cor {d.cast_strength:.1f}",
        f"    ruído           {d.noise_sigma:.1f}",
        f"    tons            de {d.black_point:.0f} a {d.white_point:.0f} (0 a 255)",
        "  etapas:",
    ]
    lines += [f"    {name:20s} {note}" for name, note in res.notes.items()]
    if res.damage is not None:
        lines.append("  detecção de danos (candidatas descartadas por teste):")
        for detector, report in res.damage.report.items():
            parts = ", ".join(f"{k}: {v}" for k, v in report.items())
            lines.append(f"    {detector:8s} {parts}")
    lines.append("  tempo (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in res.timings_ms.items()))
    from . import runtime
    lines.append(f"  IA rodando em: {runtime.device_name() if runtime.is_available() else 'indisponível (sem onnxruntime)'}")
    return "\n".join(lines)


def _post_stages_from_args(args: argparse.Namespace) -> list:
    """Etapas de IA ligadas pela linha de comando (importadas só se usadas)."""
    stages = []
    if args.rostos:                     # antes da cor: o rosto é restaurado ainda em cinza
        from .ai.face_restore import make_face_restore_stage
        stages.append(("rostos_ia", make_face_restore_stage(args.intensidade_rostos)))
    if args.colorir:
        from .ai import ddcolor
        if args.modelo_cor == "ddcolor" and ddcolor.is_available():
            stages.append(("colorizacao", ddcolor.make_ddcolor_stage(args.intensidade_cor)))
        else:
            from .ai.colorize import is_available, make_colorize_stage
            if not is_available():
                print("aviso: modelo de colorização não baixado; rode  python -m retrato_vivo modelos")
            stages.append(("colorizacao", make_colorize_stage(args.intensidade_cor)))
    if args.ampliar:
        from .ai.superres import is_available as sr_available, make_upscale_stage
        if not sr_available("espcn", 2):
            print("aviso: modelos de super-resolução não baixados; rode  python -m retrato_vivo modelos")
        stages.append(("super_resolucao", make_upscale_stage(args.modelo_sr, args.tamanho_alvo)))
    return stages


def cmd_restaurar(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    post_stages = _post_stages_from_args(args)
    out_dir = Path(args.saida)
    src_path = Path(args.entrada)
    sources = list_images(src_path) if src_path.is_dir() else [src_path]
    if args.bordas:
        from .ai import edges
        if not edges.is_available():
            print("aviso: modelo HED não baixado (o painel de bordas sai só com o Canny); "
                  "rode  python -m retrato_vivo modelos")
    failures = 0
    for src in sources:
        try:
            img = load_image(src)
        except (ValueError, FileNotFoundError) as e:
            print(f"{src.name}: IGNORADA ({e})")
            failures += 1
            continue
        res = restore(img, cfg, post_stages=post_stages)
        save_image(out_dir / f"{src.stem}_restaurada.png", res.image)
        save_image(out_dir / f"{src.stem}_antes_depois.jpg", side_by_side(res.original, res.image))
        if args.painel:
            save_image(out_dir / f"{src.stem}_painel.jpg", steps_panel(res.steps))
        if args.bordas:
            save_image(out_dir / f"{src.stem}_bordas.jpg", edges.painel_bordas(res.original, res.image))
        if args.etapas:
            for i, (name, step) in enumerate(res.steps.items()):
                save_image(out_dir / f"{src.stem}_etapas" / f"{i:02d}_{name}.png", step)
            if res.damage is not None:
                save_image(out_dir / f"{src.stem}_etapas" / "mascara.png", res.damage.mask)

        stats = res.damage.stats if res.damage else {}
        print(f"{src.name}: {sum(res.timings_ms.values()):.0f} ms, "
              f"{stats.get('riscos', 0):.0f} riscos, {stats.get('vincos', 0):.0f} vincos, "
              f"{stats.get('manchas', 0):.0f} manchas, "
              f"{stats.get('area_danificada_pct', 0):.2f}% da área")
        if args.explicar:
            print(explain(res))
    print(f"resultados em: {out_dir.resolve()}")
    return 1 if failures else 0


def cmd_envelhecer(args: argparse.Namespace) -> int:
    aged, _ = age_photo(load_image(args.entrada), AgingConfig(seed=args.semente))
    print(f"foto envelhecida salva em {save_image(args.saida, aged)}")
    return 0


def cmd_avaliar(args: argparse.Namespace) -> int:
    """Estraga uma foto boa de propósito, restaura e mede o quanto voltou ao original.

    Faz dois testes:
    1. "Não estragar": restaura a foto boa como está. O ideal é mudar pouco.
    2. "Consertar": desenha riscos e manchas, repara e compara com a original.
    """
    cfg = _config_from_args(args)
    clean, _ = resize_max_side(load_image(args.entrada), cfg.max_side)

    untouched = restore(clean, cfg)
    print("1. Foto boa passando pelo pipeline completo (quanto ela muda):")
    print(f"   PSNR {metrics.psnr(clean, untouched.image):.1f} dB | SSIM {metrics.ssim(clean, untouched.image):.3f}"
          f" | área marcada como dano por engano: {untouched.damage.stats['area_danificada_pct']:.2f}%")

    damaged, truth = age_photo(clean, AgingConfig.only_damage(seed=args.semente))
    repair_only = _config_from_args(args)      # só detectar e apagar, sem realce
    p = repair_only.preprocess
    p.remove_color_cast = p.auto_levels = p.denoise = p.clahe = p.sharpen = False
    repaired = restore(damaged, repair_only)

    scores = metrics.mask_scores(repaired.damage.mask, truth)
    print("2. Foto com riscos e manchas desenhados (detecção e reparo):")
    print(f"   detecção:   precisão {scores['precisao']:.2f} | revocação {scores['revocacao']:.2f}"
          f" | F1 {scores['f1']:.2f}")
    print(f"   danificada: PSNR {metrics.psnr(clean, damaged):.1f} dB | SSIM {metrics.ssim(clean, damaged):.3f}")
    print(f"   reparada:   PSNR {metrics.psnr(clean, repaired.image):.1f} dB | SSIM {metrics.ssim(clean, repaired.image):.3f}")

    if args.saida:
        out = Path(args.saida)
        save_image(out / "1_original.png", clean)
        save_image(out / "2_danificada.png", damaged)
        save_image(out / "3_mascara_verdadeira.png", truth)
        save_image(out / "4_mascara_detectada.png", repaired.damage.mask)
        save_image(out / "5_mapa_de_danos.png", repaired.steps["mapa_de_danos"])
        save_image(out / "6_reparada.png", repaired.image)
        save_image(out / "7_pipeline_completo_na_foto_boa.png", untouched.image)
        print(f"imagens em: {out.resolve()}")
    return 0


def cmd_vivo(args: argparse.Namespace) -> int:
    """Restaura a foto (com rostos e cor) e cria o Retrato Vivo em MP4 e GIF."""
    from . import living_portrait as lp
    from .ai.ddcolor import make_ddcolor_stage
    from .ai.face_restore import make_face_restore_stage
    img = load_image(args.entrada)
    if not args.sem_restaurar:
        img = restore(img, post_stages=[("rostos_ia", make_face_restore_stage()),
                                        ("colorizacao", make_ddcolor_stage(0.8))]).image
    portrait = lp.make_living_portrait(img, lp.LivingPortraitConfig(seconds=args.segundos))
    out = Path(args.saida)
    stem = Path(args.entrada).stem
    print(f"vídeo: {lp.save_video(portrait, out / (stem + '_vivo.mp4'))}")
    print(f"GIF:   {lp.save_gif(portrait, out / (stem + '_vivo.gif'))}")
    save_image(out / (stem + "_profundidade.png"), lp.depth_colormap(portrait.depth))
    print(f"{len(portrait.frames)} quadros em {sum(portrait.timings_ms.values()) / 1000:.1f} s ({portrait.device})")
    return 0


def cmd_viajar(args: argparse.Namespace) -> int:
    from .time_machine import ERAS, travel
    img = load_image(args.entrada)
    eras = list(ERAS) if args.epoca == "todas" else [args.epoca]
    out = Path(args.saida)
    for era in eras:
        dst = out if len(eras) == 1 and out.suffix else out / f"{Path(args.entrada).stem}_{era}.png"
        print(f"{ERAS[era].name}: {save_image(dst, travel(img, era, seed=args.semente))}")
    return 0


def cmd_modelos(args: argparse.Namespace) -> int:
    from . import models
    failed = models.download_all(args.nomes)
    if failed:
        print(f"falharam: {', '.join(failed)}. Rode de novo para tentar só os que faltam.")
    else:
        print(f"todos os modelos estão em {models.MODELS_DIR}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retrato_vivo", description="Restaurador de fotos antigas")
    sub = p.add_subparsers(dest="comando", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--sensibilidade", type=float, default=0.5,
                        help="detecção de danos, de 0 (só o óbvio) a 1 (pega tudo); padrão 0.5")
        sp.add_argument("--metodo", choices=("ia", "telea", "ns"), default="ia",
                        help="preenchimento: ia (LaMa nos buracos largos, padrão), telea ou ns")
        sp.add_argument("--max-lado", type=int, default=2048,
                        help="fotos maiores são reduzidas para este tamanho")
        sp.add_argument("--perdas-grandes", action="store_true",
                        help="também detecta rasgos e áreas brancas")
        sp.add_argument("--semente", type=int, default=42)

    r = sub.add_parser("restaurar", help="restaura um arquivo ou todas as imagens de uma pasta")
    r.add_argument("entrada")
    r.add_argument("-o", "--saida", default="saida")
    r.add_argument("--painel", action="store_true", help="salva um painel com todas as etapas")
    r.add_argument("--etapas", action="store_true", help="salva cada etapa em um arquivo")
    r.add_argument("--perspectiva", action="store_true",
                   help="foto de uma foto tirada com o celular: acha o papel, desentorta e recorta")
    r.add_argument("--bordas", action="store_true",
                   help="salva um painel de bordas Canny x HED (Deep Learning), antes e depois")
    r.add_argument("--explicar", action="store_true",
                   help="mostra o diagnóstico e o que cada etapa decidiu")
    r.add_argument("--colorir", action="store_true",
                   help="colore fotos P&B/sépia com Deep Learning (fotos coloridas passam direto)")
    r.add_argument("--intensidade-cor", type=float, default=0.8,
                   help="força da colorização, de 0 (cinza) a 1 (cor da rede); padrão 0.8")
    r.add_argument("--modelo-cor", choices=("ddcolor", "zhang"), default="ddcolor",
                   help="ddcolor (mais realista, padrão) ou zhang (mais discreto)")
    r.add_argument("--rostos", action="store_true", help="restaura os rostos com IA (GFPGAN)")
    r.add_argument("--intensidade-rostos", type=float, default=0.7,
                   help="mistura do rosto restaurado com o original, de 0 a 1; padrão 0.7")
    r.add_argument("--ampliar", action="store_true",
                   help="amplia fotos pequenas com super-resolução (Deep Learning)")
    r.add_argument("--modelo-sr", choices=("espcn", "edsr"), default="espcn",
                   help="espcn: rápido (padrão); edsr: melhor e bem mais lento")
    r.add_argument("--tamanho-alvo", type=int, default=2048,
                   help="lado maior desejado após ampliar, em pixels; padrão 2048")
    common(r)
    r.set_defaults(func=cmd_restaurar)

    a = sub.add_parser("avaliar", help="estraga uma foto boa, restaura e mede a qualidade")
    a.add_argument("entrada")
    a.add_argument("-o", "--saida", default=None)
    common(a)
    a.set_defaults(func=cmd_avaliar)

    e = sub.add_parser("envelhecer", help="estraga uma foto boa (sépia, grão, riscos, manchas)")
    e.add_argument("entrada")
    e.add_argument("-o", "--saida", default="envelhecida.png")
    e.add_argument("--semente", type=int, default=None)
    e.set_defaults(func=cmd_envelhecer)

    v = sub.add_parser("vivo", help="Retrato Vivo: a foto restaurada vira um vídeo com efeito 3D")
    v.add_argument("entrada")
    v.add_argument("-o", "--saida", default="saida")
    v.add_argument("--segundos", type=float, default=4.0, help="duração de uma volta da câmera")
    v.add_argument("--sem-restaurar", action="store_true", help="usa a foto como está, sem restaurar antes")
    v.set_defaults(func=cmd_vivo)

    from .time_machine import ERAS
    tm = sub.add_parser("viajar", help="Máquina do tempo: a foto vira foto de outra época")
    tm.add_argument("entrada")
    tm.add_argument("--epoca", choices=[*ERAS, "todas"], default="todas")
    tm.add_argument("-o", "--saida", default="saida")
    tm.add_argument("--semente", type=int, default=None)
    tm.set_defaults(func=cmd_viajar)

    m = sub.add_parser("modelos", help="baixa os modelos pré-treinados para models/")
    m.add_argument("nomes", nargs="*", help="padrão: todos")
    m.set_defaults(func=cmd_modelos)
    return p


def main(argv: list[str] | None = None) -> int:
    # Acentos corretos mesmo quando a saída é redirecionada no Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
