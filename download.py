# import os
# import urllib.request
# import tarfile
# from tqdm import tqdm # 진행바 표시용 (선택사항: pip install tqdm)

# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)

# def download_and_extract(save_path="./data"):
#     url = "http://www.openslr.org/resources/12/test-clean.tar.gz"
#     filename = "test-clean.tar.gz"
#     full_path = os.path.join(save_path, filename)

#     os.makedirs(save_path, exist_ok=True)

#     # 1. 파일 다운로드
#     print(f"다운로드 중: {url}")
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
#         urllib.request.urlretrieve(url, filename=full_path, reporthook=t.update_to)

#     # 2. 압축 해제
#     print("압축 해제 중...")
#     with tarfile.open(full_path, "r:gz") as tar:
#         tar.extractall(path=save_path)
    
#     # 3. 압축 파일 삭제 (선택사항)
#     os.remove(full_path)
    
#     print(f"완료! 파일 위치: {os.path.join(save_path, 'LibriSpeech', 'test-clean')}")

# if __name__ == "__main__":
#     # tqdm이 없다면 pip install tqdm 하거나, 위 클래스와 with 구문을 빼고 urlretrieve만 사용하세요.
#     download_and_extract("./my_raw_audio")


import os
import torchaudio

def download_librispeech_clean_test(save_path="./data"):
    print(f"다운로드 시작: {save_path} ...")
    
    # 폴더가 없으면 생성
    os.makedirs(save_path, exist_ok=True)

    # url="test-clean" 옵션으로 clean test 셋만 다운로드
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=save_path,
        url="test-clean",
        download=True
    )
    
    print("다운로드 및 압축 해제 완료!")
    print(f"저장 위치: {os.path.join(save_path, 'LibriSpeech', 'test-clean')}")

if __name__ == "__main__":
    # 원하는 경로로 변경하세요
    MY_LOCAL_PATH = "./my_librispeech"
    download_librispeech_clean_test(MY_LOCAL_PATH)