## Video Suitability Assessment
we develop a Video Training Suitability Score (VTSS)
that integrates multiple sub-metrics, allowing us to filter high-quality videos from
the original corpus.

We release a base version of the scoring model, you can download the checkpoint from [here](https://huggingface.co/Koala-36M/Training_Suitability_Assessment). To predict the VTSS of the video, you can run:

```
cd training_suitability_assessment
pip install -r requirements.txt
mkdir ckpt
huggingface-cli download --resume-download Koala-36M/Training_Suitability_Assessment --local-dir ckpt
python inference.py
```
