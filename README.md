청각 장애인을 위한 Tiny 머신러닝 기반 Warning Glasses (2024 BME CAPSTONE)
====================================================================================================================================================================
____________________________


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

----------------------------------------------------------

Edge Impulse
-----------------------------------------------------------
저희가 프로젝트에 활용한 머신러닝 플랫폼인 Edge Impulse는 TinyML 기반의 클라우드 플랫폼입니다. 
신경망 등 다양한 알고리즘을 선택하여 음성을 비롯한 이미지 등의 학습에 사용됩니다. TinyML은 임베디드 하드웨어의 낮은 성능과 저전력 마이크로컨트롤러에서 ML을 구현할 수 있도록 지원하는 기술입니다.

https://studio.edgeimpulse.com/studio/397181
아두이노 ZIP 파일: [ei-ncc-project-1-arduino-1.0.1.zip](https://github.com/user-attachments/files/15619049/ei-ncc-project-1-arduino-1.0.1.zip)


아두이노 IDE 설정
--------------------------------------------------
해당 프로젝트에 사용되는 보드는 arduino nano 33 ble nano sense 이며 보드 내에 센서가 내장되어 보드가 외부 신호를 인식할 수 있습니다.

/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <ncc-project-1_inferencing.h>

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

// Define pin for vibration motor
const int motorPin = 4;

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");

    // Set motorPin as output
    pinMode(motorPin, OUTPUT);
    digitalWrite(motorPin, LOW); // Ensure motor is off at the start

    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    ei_printf("Starting inferencing in 2 seconds...\n");

    delay(2000);

    ei_printf("Recording...\n");

    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    ei_printf("Recording done\n");

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    // print the predictions
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": \n");

    // Variable to check if horn sound is detected
    bool horn_detected = false;

    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);

        // Check if the detected label is "horn" (change "horn" to your actual label)
        if (strcmp(result.classification[ix].label, "horn") == 0 && result.classification[ix].value > 0.8) {
            horn_detected = true;
        }
    }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

    // Control the motor based on horn detection
    if (horn_detected) {
        digitalWrite(motorPin, HIGH);  // Turn on the motor
        delay(1000);                   // Run the motor for 1 second
        digitalWrite(motorPin, LOW);   // Turn off the motor
    }
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead>>1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

    if(inference.buffer == NULL) {
        return false;
    }

    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize(4096);

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        microphone_inference_end();

        return false;
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    inference.buf_ready = 0;
    inference.buf_count = 0;

    while(inference.buf_ready == 0) {
        delay(10);
    }

    return true;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif

아두이노 도식도
--------------------------------------------------
<img width="260" alt="아두이도 도식도" src="https://github.com/ohjaeeun/BME-capstone/assets/171842597/d84fd103-15cd-449e-8550-0b61560ca641">



