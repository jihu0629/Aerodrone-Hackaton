# 다중 지역 통합 학습 데이터 구성 (`data/seg_multi`)

| 출처 | 설명 | 학습 이미지 | 학습 라벨 | 검증 이미지 | 검증 라벨 |
|---|---|---|---|---|---|
| aihub | AI Hub 해안 오염물질 (국내 사진→2~4 cm/px 거리별 판, 1024) | 2,375 | 23,461 | 125 | 1,163 |
| hawaii | 하와이 항공 정사영상 칩 (2 cm/px, 640, Zenodo 8381113) | 1,109 | 7,183 | 58 | 325 |
| mgd | 문갑도 업체 정사영상 1024 타일 (3 cm/px, AI Hub 모델 의사라벨) | 21 | 27 | 1 | 1 |
| tunisia | 튀니지 TUN-MarineLitter 해변 드론 근접 (640, Roboflow CC-BY) | 1,900 | 4,077 | 100 | 205 |
| **합계** | | **5,405** | **34,748** | **284** | **1,694** |

검증은 출처별 5 % 무작위. 평가용(문갑도 칩 48 · 하와이 평가 420칩 · 튀니지 test 179)은 학습·검증 어디에도 없음.

## 통합 클래스별 라벨 수 (학습+검증)

| 클래스 | aihub | hawaii | mgd | tunisia | 합계 |
|---|---|---|---|---|---|
| styrofoam | 5,856 | 0 | 22 | 0 | 5,878 |
| styrofoam_buoy | 2,158 | 0 | 3 | 0 | 2,161 |
| buoy | 1,245 | 1,971 | 0 | 0 | 3,216 |
| net | 574 | 606 | 0 | 0 | 1,180 |
| rope | 1,460 | 134 | 0 | 0 | 1,594 |
| plastic | 3,344 | 3,740 | 1 | 2,122 | 9,207 |
| bottle | 7,045 | 0 | 2 | 0 | 7,047 |
| metal | 1,965 | 167 | 0 | 291 | 2,423 |
| glass | 977 | 0 | 0 | 413 | 1,390 |
| wood | 0 | 574 | 0 | 223 | 797 |
| tire | 0 | 281 | 0 | 0 | 281 |
| cardboard | 0 | 0 | 0 | 292 | 292 |
| other | 0 | 35 | 0 | 941 | 976 |

## 원본 클래스 → 통합 클래스 매핑

- **aihub**: Metal→metal, Plastic_Buoy→buoy, PET_Bottle→bottle, Styrofoam_Piece→styrofoam, Plastic_ETC→plastic, Glass→glass, Rope→rope, Styrofoam_Buoy→styrofoam_buoy, Styrofoam_Box→styrofoam, Net→net, Plastic_Buoy_China→buoy
- **hawaii**: buoy→buoy, unidentified object→plastic, net cloth→net, line fragment→rope, metal→metal, tire→tire, processed wood→wood, vessel→other
- **tunisia**: Cardboar→cardboard, Fabrics→other, Glass→glass, Metal→metal, Other→other, Plastic→plastic, Wood→wood
- **mgd**: AI Hub 모델 예측 클래스를 aihub 매핑으로 변환 (conf≥0.4만 라벨)