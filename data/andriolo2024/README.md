# Andriolo et al. (2024) 그림 크롭

출처: Andriolo, U., Gonçalves, G., Hidaka, M., Gonçalves, D., Gonçalves, L.M., Bessa, F., Kako, S. (2024).
Marine litter weight estimation from UAV imagery: Three potential methodologies to advance macrolitter reports.
*Marine Pollution Bulletin*, 202, 116405. https://doi.org/10.1016/j.marpolbul.2024.116405 — **CC BY 4.0**.

| 파일 | 원본 | 실제 폭 | 크롭 해상도 |
|---|---|---|---|
| crop_fig1d_left.png / right.png | Fig. 1d | 5 m | 약 1.2 cm/px |
| crop_fig3c_left.png | Fig. 3c 왼쪽 (빨간 윤곽 = 논문의 객체 분할) | 2 m | 약 0.44 cm/px |
| crop_fig3c_right.png | Fig. 3c 오른쪽 | 1 m | 약 0.29 cm/px |

논문 PDF 그림을 잘라낸 것이라 원본 정사영상(0.4 cm/px)보다 화질이 낮고, DSM(높이)은 없다.
`python scripts/15_leirosa_validation.py --photos data/andriolo2024` 에서 검출 개수 비교에만 쓴다.
