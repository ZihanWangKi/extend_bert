export PROJECT_DATA=/path/to/data/folder # root path for storing bert model & ner data, see allennlp-configs/base.jsonnet for example
export PROJECT_MODELS=/path/to/model/folder # root path for storing checkpoints, see usages below
# index of the GPU to use
GPU=$1
# path to the jsonnet config
CONFIG=$2

MODEL_PREFIX=${PROJECT_MODELS}/allennlp_models
prefix="configs/"
suffix=".jsonnet"
config_name=${2#${prefix}}
config_name=${config_name%"$suffix"}
MODEL_DIR=${MODEL_PREFIX}/${config_name}
echo "Outputting to ${MODEL_DIR}"
if [[ -d ${MODEL_DIR} ]]; then
    echo "${MODEL_DIR}" is not empty, do you want to remove it?
    select yn in "Yes" "No"; do
        case ${yn} in
            Yes ) rm -rf ${MODEL_DIR}; break;;
            No ) exit;;
        esac
    done
fi

cuda_overwrite='{"trainer":{"cuda_device":"'0'"}}'

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU} \
  allennlp train ${CONFIG} \
    -s ${MODEL_DIR} \
    --include-package allennlp-lib \
    -o=${cuda_overwrite}
