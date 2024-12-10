from .nodes.nodes import *

NODE_CLASS_MAPPINGS = {
    "HDRConversion": HDRConversion,
}
    
NODE_DISPLAY_NAME_MAPPINGS = {
    "HDRConversion": "Generate HDR image", 
}
print("\033[34mComfyUI-HDRConversion Nodes: \033[92mLoaded\033[0m")    