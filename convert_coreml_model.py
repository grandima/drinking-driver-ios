import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft 
model = ct.models.MLModel("./oldpython_model.mlpackage") 
spec = ct.utils.load_spec("./oldpython_model.mlpackage")

input = spec.description.input[0]
input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
input.type.imageType.height = 400
input.type.imageType.width = 400

ct.utils.save_spec(spec, "OldModel.mlpackage")