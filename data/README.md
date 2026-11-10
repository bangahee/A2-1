# 이미지 출처 및 전처리

- 원본: Matplotlib 배포본의 공개 예제 이미지 `grace_hopper.jpg`.
- 공식 사용 예: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html
- 배포 경로: `matplotlib/mpl-data/sample_data/grace_hopper.jpg`.
- 이 저장소에는 RGB 원본 대신 전처리한 `grace_hopper_64.npy`를 저장한다.
- 처리: NumPy로 `0.2126 R + 0.7152 G + 0.0722 B` 회색조 변환,
  중앙 정사각형 자르기, 블록 평균으로 64×64 축소, [0,1] 범위 유지.
- 재생성: `python -m scripts.prepare_image`. 최초 실행/일반 실험은
  체크인된 NPY 파일을 사용하므로 네트워크 접근이 필요 없다.
- Matplotlib은 이미지 입출력/시각화에만 사용하며, 변환과 SVD 계산은 NumPy로 한다.
- 64×64 행렬은 최대 64개의 특이값을 가진다. 요청 k=100은 유효 k=64로
  명시하고 전체 복원한다. k=50/64에서는 분해 인자 저장량이 원본보다
  커질 수 있으므로 이를 '파일 크기 압축 성공'이라고 주장하지 않는다.
