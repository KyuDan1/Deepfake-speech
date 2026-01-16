#!/usr/bin/env python3
"""
Kaggle ASVspoof2021 데이터셋 구조 확인 스크립트

kaggle datasets download --unzip 후 실제로 어떤 파일들이 생성되는지 확인합니다.
"""

import os
from pathlib import Path
from collections import defaultdict

def check_structure(target_dir: Path):
    """디렉토리 구조와 파일 유형 분석"""

    if not target_dir.exists():
        print(f"[ERROR] 디렉토리가 존재하지 않습니다: {target_dir}")
        return

    print("=" * 70)
    print(f"디렉토리 분석: {target_dir}")
    print("=" * 70)

    # 1. 최상위 항목 나열
    print("\n[1] 최상위 항목:")
    top_items = list(target_dir.iterdir())
    for item in sorted(top_items):
        if item.is_dir():
            count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  📁 {item.name}/ ({count} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.2f} MB)")

    # 2. 파일 확장자별 통계
    print("\n[2] 파일 확장자별 통계:")
    ext_stats = defaultdict(lambda: {"count": 0, "size": 0})

    for f in target_dir.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower() if f.suffix else "(no ext)"
            ext_stats[ext]["count"] += 1
            ext_stats[ext]["size"] += f.stat().st_size

    for ext, stats in sorted(ext_stats.items(), key=lambda x: -x[1]["size"]):
        size_mb = stats["size"] / (1024 * 1024)
        print(f"  {ext:15} : {stats['count']:>8} files, {size_mb:>10.2f} MB")

    # 3. 중요 파일/폴더 존재 여부 확인
    print("\n[3] 핵심 항목 존재 여부:")

    checks = {
        "flac 폴더": target_dir / "flac",
        "keys 폴더": target_dir / "keys",
        "ASVspoof2021_DF_eval 폴더": target_dir / "ASVspoof2021_DF_eval",
        "tar.gz 파일들": list(target_dir.glob("*.tar.gz")),
        "중첩된 flac 폴더": list(target_dir.glob("**/flac")),
        "중첩된 keys 폴더": list(target_dir.glob("**/keys")),
        "trial_metadata.txt": list(target_dir.glob("**/trial_metadata.txt")),
        "CM 프로토콜 파일": list(target_dir.glob("**/*CM*.txt")),
    }

    for name, path in checks.items():
        if isinstance(path, list):
            if path:
                print(f"  ✅ {name}: {len(path)}개 발견")
                for p in path[:3]:  # 최대 3개만 표시
                    print(f"      - {p.relative_to(target_dir)}")
            else:
                print(f"  ❌ {name}: 없음")
        else:
            if path.exists():
                if path.is_dir():
                    count = sum(1 for _ in path.rglob("*") if _.is_file())
                    print(f"  ✅ {name}: 존재 ({count} files)")
                else:
                    print(f"  ✅ {name}: 존재")
            else:
                print(f"  ❌ {name}: 없음")

    # 4. .flac 파일 샘플 확인
    print("\n[4] .flac 파일 샘플 (최대 5개):")
    flac_files = list(target_dir.rglob("*.flac"))[:5]
    if flac_files:
        for f in flac_files:
            print(f"  - {f.relative_to(target_dir)}")
    else:
        print("  (없음)")

    # 5. 결론
    print("\n" + "=" * 70)
    print("[결론]")

    has_flac_direct = (target_dir / "flac").exists()
    has_flac_nested = bool(list(target_dir.glob("**/flac")))
    has_tar_gz = bool(list(target_dir.glob("*.tar.gz")))
    has_flac_files = bool(list(target_dir.rglob("*.flac")))

    if has_flac_files and (has_flac_direct or has_flac_nested):
        print("✅ Kaggle --unzip이 모든 압축을 풀어 flac 파일이 바로 사용 가능합니다!")
        print("   → download_datasets.py의 tar.gz 추출 코드는 불필요할 수 있습니다.")
    elif has_tar_gz and not has_flac_files:
        print("⚠️  Kaggle --unzip이 최상위 zip만 풀었습니다. 내부 tar.gz가 남아있습니다.")
        print("   → download_datasets.py의 tar.gz 추출 코드가 필요합니다.")
    elif has_flac_files:
        print("✅ flac 파일이 존재합니다.")
    else:
        print("❓ 예상치 못한 구조입니다. 위 정보를 확인해주세요.")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle ASVspoof2021 구조 확인")
    parser.add_argument("--path", "-p",
                        default="/mnt/tmp/Deepfake-speech/data/ASVspoof2021",
                        help="확인할 디렉토리 경로")
    args = parser.parse_args()

    check_structure(Path(args.path))
