#!/bin/bash
echo -e "${INFO}Download Model From HuggingFace-Mirror"
PRETRINED_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
G2PW_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
UVR5_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
NLTK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
PYOPENJTALK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
FUN_ASR_URL="https://cvws.icloud-content.com/B/AXBceAVMuVklfTrAtRBssiar-Xi9Aba7AUBt6R62PK45_MDIBtdk6mm4/models.zip?o=As2GIKdmUuj6RnHNCDMfbmfZUoenpmEdcXZYXinGfYaO&v=1&x=3&a=CAogs1WUqCw8fFbdcZdI9tPOI96o5Cj0rCWZ3eSFW8QJF2oSbxCJ8MD0qTMYic2c9qkzIgEAUgSr-Xi9WgRk6mm4aifN_PU2dUB5c-iJ1ziZXUkg-tDF_n-NmbvYBapb6Qrslpn4KtlUp8hyJ1I8CNDC6lqvMJuOHelPz_4PN0xiEPBP-D5uw0DM7KfLDZkBlHYIuw&e=1763600443&fl=&r=78a8e895-bf87-4cfd-805e-49bad4cb20df-1&k=ViRer5obYscfo1Ic510s8g&ckc=com.apple.clouddocs&ckz=com.apple.CloudDocs&p=104&s=tlwf84XHXe77B7jfXMknqkm5jAk&+=25f1faaf-4821-44d1-8462-ddc834a8b9f8"

BASE_DIR="models"

run_wget_quiet() {
    if wget --tries=25 --wait=5 --read-timeout=40 -q --show-progress "$@" 2>&1; then
        tput cuu1 && tput el
    else
        echo -e "${ERROR} Wget failed"
        exit 1
    fi
}

if [ ! -d "${BASE_DIR}/GPT_SoVITS/pretrained_models/sv" ]; then
    echo -e "${INFO}Downloading Pretrained Models..."
    rm -rf pretrained_models.zip
    run_wget_quiet "$PRETRINED_URL"

    unzip -q -o pretrained_models.zip -d ${BASE_DIR}/GPT_SoVITS
    rm -rf pretrained_models.zip
    echo -e "${SUCCESS}Pretrained Models Downloaded"
else
    echo -e "${INFO}Pretrained Model Exists"
    echo -e "${INFO}Skip Downloading Pretrained Models"
fi

if [ ! -d "${BASE_DIR}/GPT_SoVITS/text/G2PWModel" ]; then
    echo -e "${INFO}Downloading G2PWModel.."
    rm -rf G2PWModel.zip
    run_wget_quiet "$G2PW_URL"

    unzip -q -o G2PWModel.zip -d ${BASE_DIR}/GPT_SoVITS/text
    rm -rf G2PWModel.zip
    echo -e "${SUCCESS}G2PWModel Downloaded"
else
    echo -e "${INFO}G2PWModel Exists"
    echo -e "${INFO}Skip Downloading G2PWModel"
fi

UVR5_MODELS_DIR="${BASE_DIR}/uvr5"
if find -L "${UVR5_MODELS_DIR}" -mindepth 1 ! -name '.gitignore' | grep -q .; then
    echo -e"${INFO}UVR5 Models Exists"
    echo -e "${INFO}Skip Downloading UVR5 Models"
else
    echo -e "${INFO}Downloading UVR5 Models..."
    rm -rf uvr5_weights.zip
    run_wget_quiet "$UVR5_URL"
    mkdir -p "${UVR5_MODELS_DIR}"
    unzip -q -o uvr5_weights.zip -d ${UVR5_MODELS_DIR}
    rm -rf uvr5_weights.zip
    echo -e "${SUCCESS}UVR5 Models Downloaded"
fi


ASR_MODELS_DIR="${BASE_DIR}/asr"
if find -L "${ASR_MODELS_DIR}" -mindepth 1 ! -name '.gitignore' | grep -q .; then
    echo -e"${INFO}ASR Models Exists"
    echo -e "${INFO}Skip Downloading ASR Models"
else
    echo -e "${INFO}Downloading ASR Models..."
    rm -rf models.zip
    run_wget_quiet -O "models.zip" "$FUN_ASR_URL"
    mkdir -p "${ASR_MODELS_DIR}"

    unzip -q -o models.zip -d ${ASR_MODELS_DIR}
    rm -rf models.zip
    echo -e "${SUCCESS}ASR Models Downloaded"
fi
