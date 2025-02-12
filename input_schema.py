INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["a living room, bright modern Scandinavian style house, large windows, magazine photoshoot, 8k, studio lighting"]
    },
    "negative_prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["low quality"]
    }
    ,
    "num_inference_steps": {
        'datatype': 'INT8',
        'required': True,
        'shape': [1],
        'example': [28]
    }
    ,
    "guidance_scale": {
        'datatype': 'FP32',
        'required': True,
        'shape': [1],
        'example': [7.0]
    }
}