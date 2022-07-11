# AI Model
- [x] jasper

### 개발 환경
- PyTorch 1.11.3, CUDA 11.3, Python 3.8 and Ubuntu18.04
- dependency
```
pip install -r requirements.txt
```
- [apex 설치](https://github.com/NVIDIA/apex)
- beamsearch_decoder 설치
```
* cd /hypersp/hypersp/modules/ 
* ./install_beamsearch_decoders.sh
```

### 회의 음성 데이터
[회의 음성](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=132) AI Hub 데이터셋을 Training 데이터만 다운 <br>

valid, test 데이터셋은 나스에 저장되어 있는 데이터 사용 <br>
path :  \\172.30.1.24\데이터사업추진팀\01. 프로젝트\23. 2022년 AI바우처 지원사업\감사nlp\stt\data <br>

pre-trained model <br>
path : \\172.30.1.24\데이터사업추진팀\01. 프로젝트\23. 2022년 AI바우처 지원사업\감사nlp\stt\model


```
* 회의 음성 데이터 학습
  * 학습데이터셋을 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/kconfspeech
  * scripts/preprocess_kconfspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_kconfspeech.sh
  * scripts/train.sh에서 전처리된 파일 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/train.sh
  
* 회의 음성 데이터 평가
  * 학습데이터셋 및 모델 checkpoint를 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/kconfspeech
  * scripts/preprocess_kconfspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_kconfspeech.sh
  * scripts/evaluation.sh에서 전처리된 파일 경로, checkpoint 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/evaluation.sh
```

## DEMO
* demo 폴더에서 5가지의 데모 서버를 제공합니다
  * 발화 단위 대화록 생성 서비스 실행방법
    * cd /hypersp
    * python demo/minutes_demo.py
    * port: 8888

* NOTICE
  * local이 아닌 곳에서 데모를 띄울 경우 아래 파일에서 ip 주소를 알맞게 변경해주셔야 합니다
    * 발화 단위 대화록 생성 서비스: demo/templates/index_minutes.html

## AI 허브 개방 데이터
```
- AI 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI 응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.
- 본 AI데이터 등은 인공지능 기술 및 제품·서비스 발전을 위하여 구축하였으며, 지능형 제품・서비스, 챗봇 등 다양한 분야에서 영리적・비영리적 연구・개발 목적으로 활용할 수 있습니다.
```

