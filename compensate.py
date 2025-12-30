import argparse
import json
from pathlib import Path

from weight_compensation import CompensationModel, DeviceCalibration


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="用 20°C 标定参数 (T20,S0,S100) 和全局温度补偿模型，对芯片重量读数做温漂修正。",
    )
    p.add_argument("--model", type=str, required=True, help="模型 JSON 路径，例如 models/model.json")
    p.add_argument("--t20", type=float, required=True, help="实际 20°C 时的芯片温度读数 (T20)")
    p.add_argument("--s0", type=float, required=True, help="20°C 空载/去皮后的芯片重量读数 (S0)")
    p.add_argument("--s100", type=float, required=True, help="20°C 放 100g 时的芯片重量读数 (S100)")
    p.add_argument("--t", type=float, required=True, help="当前芯片温度读数 (T)")
    p.add_argument("--s", type=float, required=True, help="当前芯片重量读数/信号 (S)")
    p.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return p


def main() -> None:
    args = build_parser().parse_args()

    model = CompensationModel.load_json(args.model)
    cal = DeviceCalibration(t20_chip=args.t20, s0=args.s0, s100=args.s100)

    s_corr = float(model.compensate_signal(s_raw=args.s, t_chip=args.t, calibration=cal))

    if args.json:
        payload = {
            "s_raw": args.s,
            "t_chip": args.t,
            "t20_chip": args.t20,
            "s0": args.s0,
            "s100": args.s100,
            "s_corrected": s_corr,
        }
        print(json.dumps(payload, ensure_ascii=False))
        return

    print(f"s_raw={args.s}, t_chip={args.t} -> s_corrected={s_corr:.6f}")


if __name__ == "__main__":
    main()

