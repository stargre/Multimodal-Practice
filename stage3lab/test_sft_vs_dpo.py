import os
import json
import time
import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from transformers import CLIPImageProcessor


from models.qwen25_fork.build_model import build_model
from models.qwen25_fork.processing_hybrid import HybridQwen25Processor



# ============================================================
# Path
# ============================================================


SFT_CKPT = (
    "./runs/sft_150k_ddp_6gpu/best_model.pt"
)

DPO_CKPT = (
    "./runs/dpo_mmrlhf_6gpu/best_model.pt"
)


VAL_JSON = (
    "./data_splits/mmrlhf_dpo/val.json"
)


OUT_DIR = Path(
    "./runs/sft_vs_dpo_eval"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_JSON = OUT_DIR / "sft_vs_dpo_result.json"
LOG_FILE = OUT_DIR / "sft_vs_dpo_test.log"



# ============================================================
# Model paths
# ============================================================


QWEN25_PATH = "./checkpoints/Qwen2.5-VL-3B-Instruct"

VIT_PATH = "./checkpoints/InternViT-300M-448px-V2_5"

LLM_PATH = "./checkpoints/Qwen3-0.6B-Base"



DEVICE = "cuda"

DTYPE = torch.bfloat16



# ============================================================
# logging
# ============================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            mode="w",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger()



# ============================================================
# checkpoint loading
# ============================================================


def load_project_model(
        checkpoint_path,
        name
):

    logger.info("")
    logger.info(
        "=============================="
    )

    logger.info(
        "Loading %s model",
        name
    )


    model, tokenizer = build_model(
        device=DEVICE,
        dtype=DTYPE,
        qwen25_path=QWEN25_PATH,
        vit_path=VIT_PATH,
        llm_path=LLM_PATH,
    )


    logger.info(
        "Loading checkpoint: %s",
        checkpoint_path
    )


    state = torch.load(
        checkpoint_path,
        map_location="cpu"
    )


    logger.info(
        "Checkpoint type: %s",
        type(state)
    )


    missing, unexpected = model.load_state_dict(
        state,
        strict=False
    )


    logger.info(
        "Missing keys: %d",
        len(missing)
    )

    logger.info(
        "Unexpected keys: %d",
        len(unexpected)
    )


    if len(missing) > 0:

        logger.info(
            "First missing keys:"
        )

        for k in missing[:10]:
            logger.info(
                "  %s",
                k
            )


    model.eval()


    return model, tokenizer



# ============================================================
# Processor
# ============================================================


def build_processor(tokenizer):


    image_processor = CLIPImageProcessor.from_pretrained(
        VIT_PATH
    )


    processor = HybridQwen25Processor(
        tokenizer,
        image_processor,
        tile_size=448,
        max_tiles_per_side=2,
    )


    return processor



# ============================================================
# device
# ============================================================


def move_batch(batch):

    output={}

    for k,v in batch.items():

        if isinstance(v,torch.Tensor):

            output[k]=v.to(
                DEVICE
            )

        else:

            output[k]=v


    return output



# ============================================================
# Generation
# ============================================================


@torch.inference_mode()
def generate_answer(
        model,
        processor,
        image,
        question
):


    prompt = processor.user_prompt(
        question,
        num_images=1
    )


    batch = processor.encode_images(
        [prompt],
        [[image]]
    )


    batch = move_batch(batch)


    # remove None fields
    batch = {
        k:v
        for k,v in batch.items()
        if v is not None
    }


    model.model.rope_deltas=None


    start=time.time()


    output_ids = model.generate(
        **batch,
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,
        eos_token_id=[
            processor.tokenizer.eos_token_id,
            processor.tokenizer.convert_tokens_to_ids(
                "<|im_end|>"
            )
        ],
        pad_token_id=(
            processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id
        ),
        use_cache=True,
    )


    cost=time.time()-start



    input_len=batch["input_ids"].shape[1]


    generated_ids = output_ids[:,input_len:]


    text = processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]


    return text.strip(), generated_ids.shape[1], cost



# ============================================================
# Main
# ============================================================


def main():


    logger.info(
        "========== SFT VS DPO TEST =========="
    )


    logger.info(
        "torch=%s",
        torch.__version__
    )



    # -----------------------------
    # Load models
    # -----------------------------


    sft_model, sft_tokenizer = load_project_model(
        SFT_CKPT,
        "SFT"
    )


    dpo_model, dpo_tokenizer = load_project_model(
        DPO_CKPT,
        "DPO"
    )


    processor = build_processor(
        sft_tokenizer
    )



    # -----------------------------
    # Load dataset
    # -----------------------------


    logger.info(
        "Loading dataset:"
    )

    logger.info(
        VAL_JSON
    )


    with open(
        VAL_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        dataset=json.load(f)



    logger.info(
        "Samples: %d",
        len(dataset)
    )



    results=[]



    # -----------------------------
    # inference
    # -----------------------------


    for idx,item in enumerate(
        tqdm(dataset)
    ):


        image_path=item["image"]

        question=item["prompt"]


        if not os.path.exists(image_path):

            logger.warning(
                "Missing image %s",
                image_path
            )

            continue



        image=Image.open(
            image_path
        ).convert("RGB")



        logger.info("")
        logger.info(
            "================================"
        )

        logger.info(
            "Sample %d",
            idx
        )

        logger.info(
            "ID: %s",
            item["id"]
        )

        logger.info(
            "Question:"
        )

        logger.info(
            question
        )



        # SFT

        sft_answer,sft_len,sft_time = generate_answer(
            sft_model,
            processor,
            image,
            question
        )


        logger.info(
            ""
        )

        logger.info(
            "[SFT]"
        )

        logger.info(
            sft_answer
        )



        # DPO


        dpo_answer,dpo_len,dpo_time = generate_answer(
            dpo_model,
            processor,
            image,
            question
        )


        logger.info(
            ""
        )

        logger.info(
            "[DPO]"
        )

        logger.info(
            dpo_answer
        )



        results.append(
            {
                "id":item["id"],
                "image":image_path,
                "prompt":question,

                "chosen":item.get(
                    "chosen",
                    ""
                ),

                "rejected":item.get(
                    "rejected",
                    ""
                ),

                "sft_answer":sft_answer,

                "dpo_answer":dpo_answer,

                "sft_length":int(sft_len),

                "dpo_length":int(dpo_len),

                "sft_time":float(sft_time),

                "dpo_time":float(dpo_time),
            }
        )



        # save incrementally
        with open(
            RESULT_JSON,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2
            )



    logger.info("")
    logger.info(
        "Finished."
    )

    logger.info(
        "Saved json: %s",
        RESULT_JSON
    )

    logger.info(
        "Saved log: %s",
        LOG_FILE
    )



if __name__=="__main__":

    main()