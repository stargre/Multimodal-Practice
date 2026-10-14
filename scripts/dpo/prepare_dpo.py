import json
import random
from pathlib import Path
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True
    )

    parser.add_argument(
        "--out",
        required=True
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=2000
    )

    args = parser.parse_args()


    root = Path(args.root)


    samples=[]

    stats={
        "total":0,
        "image":0,
        "video":0,
        "missing":0,
        "same":0
    }


    with open(
        root/"dpo_pairs.jsonl"
    ) as f:

        for line in f:

            x=json.loads(line)

            stats["total"] +=1


            # skip video
            if "video" in x:
                stats["video"]+=1
                continue


            if "image" not in x:
                continue


            stats["image"]+=1


            img=root/x["image"]


            if not img.exists():

                stats["missing"]+=1

                continue



            chosen=x["chosen"].strip()

            rejected=x["rejected"].strip()



            if chosen == rejected:

                stats["same"]+=1

                continue



            sample={

                "image":str(img),

                "prompt":x["prompt"],

                "chosen":chosen,

                "rejected":rejected,

                "reference_chosen_logp":
                    x.get(
                    "reference_chosen_logp"
                    ),

                "reference_rejected_logp":
                    x.get(
                    "reference_rejected_logp"
                    ),

                "id":x["id"]

            }


            samples.append(sample)



    print("====================")
    print(stats)
    print("====================")

    print(
        "final samples:",
        len(samples)
    )



    random.seed(42)

    random.shuffle(samples)


    val=samples[:args.val_size]

    train=samples[args.val_size:]



    out=Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True
    )


    json.dump(
        train,
        open(
            out/"train.json",
            "w"
        ),
        indent=2
    )


    json.dump(
        val,
        open(
            out/"val.json",
            "w"
        ),
        indent=2
    )


    print(
        "train:",
        len(train)
    )

    print(
        "val:",
        len(val)
    )



if __name__=="__main__":

    main()