청각 장애인을 위한 Tiny 머신러닝 기반 warning glasses (2024 BME CAPSTONE)
-----------------------------------------------------------------------------------------------

-----------------------------

프로젝트 목적
-----------------

저희가 만들고자 하는 주된 이유는 청각 장애인들에게 어플리케이션의 사용 없이 언제나 위험 상황을 감지할 수 있게 하며, 애플워치 등과 같은 관련 제품들과 비교하였을 때 경제적으로 저렴하고 편하게 착용할 수 있는 부분을 고려하였을 때 이 제품이 적정하다고 생각하였습니다.

프로젝트 도식도 & 설계도
------------------------------------------------
저희가 진행하려는 프로젝트의 전체적인 흐름을 살펴보면 part 1에서 edge impulse라는 머신러닝 개발 플랫폼을 활용하여 다음과 같은 순서로 진행하였고,
Part2에서는 모델이 아두이노에 내장되어 소리 자동 감지 및 진동 모터 작동을 위한 코드 구현을 진행하고 있습니다.
![image](https://github.com/ohjaeeun/BME-capstone/assets/129700005/90a9a5e9-ba88-40b1-bdb0-c73c312e4a1c)

----------------------------------------------------
제품에 대한 예상 결과입니다.
![image](https://github.com/ohjaeeun/BME-capstone/assets/129700005/d11449fc-c7ea-4024-910d-34c3c0192fbf)

Edge Impulse
-----------------------------------------------------------
저희가 프로젝트에 활용한 머신러닝 플랫폼인 Edge Impulse는 TinyML 기반의 클라우드 플랫폼입니다. 
신경망 등 다양한 알고리즘을 선택하여 음성을 비롯한 이미지 등의 학습에 사용됩니다. TinyML은 임베디드 하드웨어의 낮은 성능과 저전력 마이크로컨트롤러에서 ML을 구현할 수 있도록 지원하는 기술입니다.

아두이노 IDE 설정
--------------------------------------------------
해당 프로젝트에 사용되는 보드는 arduino nano 33 ble nano sense 이며 보드 내에 센서가 내장되어 보드가 외부 신호를 인식할 수 있습니다.


