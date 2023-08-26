import argparse
import torch
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from torch.utils.mobile_optimizer import optimize_for_mobile
import os

parser=argparse.ArgumentParser()
parser.add_argument("--source_path",required=True,help="Path to the trained model ckpt file.")
parser.add_argument("--target_path",required=True,help="Path where to export the trained model.")
parser.add_argument("--optimize_for_mobile",default=True,required=False,help="Whether to apply optimization for mobile.")


args = parser.parse_args()


#Load the model.
parseq = load_from_checkpoint(args.source_path).eval()

#Dynamic quantize model.
quantized_parseq = torch.quantization.quantize_dynamic(
    parseq, {torch.nn.Linear}, dtype=torch.qint8
)

#Convert to torchscript.
dummy_tensor=torch.rand((1,3,parseq.hparams.img_size[0],parseq.hparams.img_size[1]))
torchscript_model=torch.jit.trace(quantized_parseq,dummy_tensor)

#Optimize for mobile devices if required.

if args.optimize_for_mobile:
	torchscript_model_optimized=optimize_for_mobile(torchscript_model)
else:
	torchscript_model_optimized=torchscript_model

#Save the optimized model.
target_path=args.target_path
target_path=os.path.join(target_path,"optimized_model.pt")
torch.jit.save(torchscript_model_optimized,"/content/parseq_optimized.pt")

