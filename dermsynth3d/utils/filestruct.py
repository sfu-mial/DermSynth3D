import abc


def custom_directories(user: str):
    if user == "jer":
        return JerFolders()
    elif user == "arezou":
        return ArezouFolders()
    elif user == "ashish":
        return AshishFolders()
    else:
        raise ValueError("Unknown user = {}".format(user))


class FolderStructure(abc.ABC):
    @abc.abstractmethod
    def backgrounds(self):
        """Directory to the background images.

        Currently we use the "Bedroom" dataset from here.
        https://www.kaggle.com/robinreni/house-rooms-image-dataset

        You'll have to download and update this path.
        """
        pass

    @abc.abstractmethod
    def bodytex_highres(self):
        """Directory of the 3dBodyTex high-res meshes."""
        pass

    @abc.abstractmethod
    def new_textures(self):
        """Directory where the modified textures are saved.

        Assumes the nonskin texture masks are also stored at this location.

        This is used in `blend_locations.ipynb`,
        which saves the pasted lesions in the texture image
        and the corresponding texture masks.

        As well, in `blend3d.ipynb` to store the blended texture images.
        """
        pass

    @abc.abstractmethod
    def anatomy(self):
        """Directory storing the vertices to anatomy labels.

        Ashish used the approach from uni.lu to determine anatomical labels
        for each 3dbodytex high resolution mesh.
        Download from:
        https://drive.google.com/drive/folders/1KuQ1Ttax2DXbe1vsB5quVLAZ1wyhho5B?usp=sharing
        """
        pass

    @abc.abstractmethod
    def fitzpatrick17k(self):
        """Directory storing the fitz17k images."""
        pass

    @abc.abstractmethod
    def fitzpatrick17k_annotations(self):
        """Directory storing the manual annotations for select fitz17k images.

        If you don't have access, request access from Jer.

        Download the `annotations` directory and subdirectories from:
        https://drive.google.com/drive/folders/1d7Nv3w7ewCfutY4rdqOJDZhjiDmyPDxZ?usp=sharing
        """
        pass

    def __str__(self):
        """Returns the folder names"""
        return "\n".join(
            [
                self.backgrounds(),
                self.bodytex_highres(),
                self.new_textures(),
                self.anatomy(),
                self.fitzpatrick17k(),
                self.fitzpatrick17k_annotations(),
            ]
        )


class JerFolders(FolderStructure):
    def __init__(self):
        pass

    def backgrounds(self):
        return "/mnt/d/data/archive/House_Room_Dataset/Bedroom/"

    def bodytex_highres(self):
        return "/mnt/d/data/3dbodytex-1.1-highres/3dbodytex-1.1-highres/"

    def new_textures(self):
        return "/mnt/d/data/3dbodytex-1.1-highres/3DBodyTex_nonskinfaces/3DBodyTex_nonskinfaces"
        # return '/mnt/d/data/3dbodytex-1.1-highres/lesions'

    def anatomy(self):
        return "/mnt/d/data/bodytex/bodytex_anatomy/bodytex_anatomy_labels"

    def fitzpatrick17k(self):
        return "/mnt/d/data/fitzpatrick17k/data/finalfitz17k"

    def fitzpatrick17k_annotations(self):
        return "/mnt/d/data/fitzpatrick17k/annotations/annotations-20220429T234131Z-001/annotations"


class ArezouFolders(FolderStructure):
    def __init__(self):
        pass

    def backgrounds(self):
        return (
            "../../../3DBlended/SyntheticData/Backgrounds/House_Room_Dataset/Bedroom/"
        )

    def bodytex_highres(self):
        return "../../../3DBodyTex/3dbodytex-1.1-highres/"

    def new_textures(self):
        return "../../../3DBlended/SyntheticData/Meshes/selected"

    def anatomy(self):
        return "../../../3DBlended/bodytex_anatomy/bodytex_anatomy_labels"

    def fitzpatrick17k(self):
        return "../../../../skin_fairness/data/fitz17k/images/all"

    def fitzpatrick17k_annotations(self):
        print("Warning! Update this with your path")
        return None


class AshishFolders(FolderStructure):
    def __init__(self):
        pass

    def backgrounds(self):
        return "../data/background/House_Room_Dataset/Bedroom"

    def bodytex_highres(self):
        return "../data/3dbodytex-1.1-highres/"

    def new_textures(self):
        return "../data/lesions/"

    def anatomy(self):
        return "../data/bodytex/bodytex_anatomy/bodytex_anatomy_labels/"

    def fitzpatrick17k(self):
        return "./data/annotations/fitz17k/finalfitz17k"

    def fitzpatrick17k_annotations(self):
        print("Warning! Update this with your path")
        return None
