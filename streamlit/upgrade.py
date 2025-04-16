import pkg_resources
import subprocess

# 1. 기존 requirements.txt에서 패키지 이름만 추출
with open("requirements.txt", "r") as f:
    original_lines = f.readlines()

package_names = []
for line in original_lines:
    line = line.strip()
    if line and not line.startswith("#"):
        pkg = line.split("==")[0].split(">=")[0]
        package_names.append(pkg.lower())

# 2. 현재 설치된 패키지 목록 가져오기
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# 3. 원래 있던 패키지의 현재 버전으로 새 requirements 생성
updated_lines = []
for name in package_names:
    updated_lines.append(f"{name}=={installed_packages[name]}")

# 4. 덮어쓰기
with open("requirements.txt", "w") as f:
    f.write("\n".join(updated_lines) + "\n")

print("✅ 기존 requirements.txt 기반으로 최신 버전으로 갱신 완료!")