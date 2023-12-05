import os
import sys
import argparse

import torch
from pprint import pprint

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir, "skin3d")
    )
)
from dermsynth3d.utils.utils import yaml_loader
from dermsynth3d import BlendLesions, Generate2DViews, SelectAndPaste
# combine all yamls into one
default = yaml_loader("configs/default.yaml")
main = yaml_loader("configs/blend.yaml")
main.update(default)
renderer = yaml_loader("configs/renderer.yaml")
main.update(renderer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh_name", "-m", default=None, type=str, help="Name of mesh"
    )
    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate for optimization"
    )
    parser.add_argument(
        "--percent_skin",
        "-ps",
        default=0.1,
        type=float,
        help="skin threshold for saving the view",
    )
    parser.add_argument(
        "--num_iter",
        "-i",
        default=None,
        type=int,
        help="number of iterations for blending",
    )
    parser.add_argument(
        "--num_views", "-v", default=None, type=int, help="number of views to generate"
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        default=None,
        type=str,
        help="path to save the generated views",
    )
    parser.add_argument(
        "--num_paste",
        "-n",
        default=None,
        type=int,
        help="number of lesions to paste per mesh",
    )
    parser.add_argument(
        "--paste", action="store_true", help="whether to force pasting or not"
    )
    parser.add_argument(
            "--view_size",
            "-vs",
            default=512,
            type=int,
            help="size of the generated views",
            )
    parser.add_argument(
            "--device",
            "-d",
            default=None,
            type=str,
            help="device to run the code on",
            )
    parser.add_argument(
            "--location",
            "-l",
            action="store_true",
            help="whether to only run select_locations()", 
            )
    parser.add_argument(
            "--blend",
            "-b",
            action="store_true",
            help="whether to only run blend_lesions()", 
            )
    parser.add_argument(
            "--generate",
            "-g",
            action="store_true",
            help="whether to only run synthesize_views()", 
            )
    

    args = parser.parse_args()
    args.view_size = str((args.view_size, args.view_size))
    args = vars(args)
    for key in args:
        if args[key] is not None:
            if key in main["blending"]:
                main["blending"][key] = args[key]
            if key in main["generate"]:
                main["generate"][key] = args[key]

    if args["device"] is not None:
        if args["device"] == "cpu":
            device = torch.device("cpu")
        elif args["device"] == "cuda":
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            raise ValueError("Invalid device: {}".format(args["device"]))
    else:
       # Setup device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
 
    print ("*********************************")
    print ("Running on device: {}".format(device))
    print ("Running on mesh: {}".format(main["blending"]["mesh_name"]))
    print ("*********************************\n")
    if args["location"] and not args["blend"] and not args["generate"]:
        print ("\nRunning only select_locations()")
        print ("*********************************")
        print (f"Looking for suitable locations to paste {main['blending']['num_paste']} lesions..")
        locations = SelectAndPaste(config=main, device=device)
        locations.paste_on_locations()
        print ("\nDone selecting locations...")
        print ("*********************************\n")
        sys.exit()
    if args["blend"] and not args["location"] and not args["generate"]:
        print ("\nRunning only blend_lesions()")
        print ("*********************************")
        print ("Blending lesions...")
        print ("Blending for {} iterations".format(main["blending"]["num_iter"]))
        print ("Blending with {} learning rate".format(main["blending"]["lr"]))
        blender = BlendLesions(config=main, device=device)
        blender.blend_lesions()
        print ("\nDone blending lesions...")
        print ("*********************************\n")
        sys.exit()
    if args["generate"] and not args["location"] and not args["blend"]:
        print ("\nRunning only synthesize_views()")
        print ("*********************************")
        print ("Generating 2D views...")
        print ("Generating {} views".format(main["generate"]["num_views"]))
        print ("Generating views with {} skin threshold".format(main["generate"]["percent_skin"]))
        print (f"Saving views at {main['generate']['save_dir']} with size {main['generate']['view_size']}")
        renderer = Generate2DViews(config=main, device=device)
        renderer.synthesize_views()
        print ("\nDone for mesh: {}".format(main["blending"]["mesh_name"]))
        print ("*********************************\n")
        exit()
    

    print ("*********************************")
    print (f"\nLooking for suitable locations to paste {main['blending']['num_paste']} lesions..")
    locations = SelectAndPaste(config=main, device=device)
    locations.paste_on_locations()
    print ("\nDone pasting lesions...")
    print ("*********************************")
    print ("\nBlending lesions...")
    print ("Blending with {} iterations".format(main["blending"]["num_iter"]))
    print ("Blending with {} learning rate".format(main["blending"]["lr"]))
    blender = BlendLesions(config=main, device=device)
    blender.blend_lesions()
    print ("\nDone blending lesions...")
    print (f"Saving texure maps with blended lesions at {main['blending']['tex_dir']}")
    print ("*********************************")
    print ("\nGenerating 2D views...")
    print ("Generating {} views".format(main["generate"]["num_views"]))
    print ("Generating views with {} skin threshold".format(main["generate"]["percent_skin"]))
    print (f"Saving views at {main['generate']['save_dir']} with size {main['generate']['view_size']}")
    renderer = Generate2DViews(config=main, device=device)
    renderer.synthesize_views()
    print ("\nDone for mesh: {}".format(main["blending"]["mesh_name"]))
