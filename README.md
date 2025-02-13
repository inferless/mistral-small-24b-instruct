# Tutorial - Deploy Stable-Diffusion-3 using Inferless
[Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) sets a new benchmark in AI image generation, delivering unparalleled image quality with enhanced efficiency. Utilizing a sophisticated Multimodal Diffusion Transformer (MMDiT) architecture, it significantly reduces noise and improves clarity.

## TL;DR:
- Deployment of stable-diffusion-3-medium-diffusers model using [Diffusers](https://github.com/huggingface/diffusers).
- You can expect an average latency of `4.4 sec` for generating an image in `28`steps. This setup has an average cold start time of `9.9 sec` *(For benchmark we have used Nvidia A100-80GB GPU)*.
- Dependencies defined in `inferless-runtime-config.yaml`.
- GitHub/GitLab template creation with `app.py`, `inferless-runtime-config.yaml` and `inferless.yaml`.
- Model class in `app.py` with `initialize`, `infer`, and `finalize` functions.
- Custom runtime creation with necessary system and Python packages.
- Model import via GitHub with `input_schema.py` file.
- Recommended GPU: NVIDIA A10.
- Custom runtime selection in advanced configuration.
- Final review and deployment on the Inferless platform.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the inferless-runtime.yaml file given above. Finally, ensure you save your changes by clicking on the save button.

### Add Your Hugging Face Access Token
Go into the `inferless.yaml` and replace `<hugging_face_token>` with your hugging face access token. Make sure to check the repo is private to protect your hugging face token.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and select your provider, and use the forked repo URL as the **Model URL**.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
          --header 'Content-Type: application/json' \
          --header 'Authorization: Bearer <your_api_key>' \
          --data '{
              "inputs": [
                {
                  "data": ["a living room, bright modern Scandinavian style house, large windows, magazine photoshoot, 8k, studio lighting"],
                  "name": "prompt",
                  "shape": [1],
                  "datatype": "BYTES"},
                {
                  "data": ["low quality"],
                  "name": "negative_prompt",
                  "shape": [1],
                  "datatype": "BYTES"},
                {
                  "data": [28],
                  "name": "num_inference_steps"
                  "shape": [1],
                  "datatype": "INT8"},
                {
                  "data": [7.0],
                  "name": "guidance_scale"
                  "shape": [1],
                  "datatype": "FP32"}
                  ]
            }'
```


---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](https://docs.inferless.com/model-import/input-output-schema) for more.

```python
def infer(self, inputs):
    prompt = inputs["prompt"]
    negative_prompt = inputs["negative_prompt"]
    inference_steps = inputs["num_inference_steps"]
    guidance_scale = inputs["guidance_scale"]
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.
```python
def finalize(self):
    self.pipe = None
```

For more information refer to the [Inferless docs](https://docs.inferless.com/).
