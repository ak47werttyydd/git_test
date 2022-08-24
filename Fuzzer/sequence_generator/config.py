import numpy as np
test_passes = np.array(
    ['InferType', 'AlterOpLayout', 'AnnotateSpans', 'BackwardFoldScaleAxis', 'BatchingOps', 
    'CanonicalizeCast', 'CanonicalizeOps', 'CombineParallelBatchMatmul', 'CombineParallelConv2D', 'CombineParallelDense', 
    'DeadCodeElimination', 'DefuseOps', 'DynamicToStatic', 'EliminateCommonSubexpr', 'FastMath', 'FoldConstant', 
    'FoldExplicitPadding', 'FoldScaleAxis', 'ForwardFoldScaleAxis', 'FuseOps', 'Inline', 'LambdaLift', 'Legalize', 'MergeCompilerRegions', 
    'PartitionGraph', 'RemoveUnusedFunctions', 'SimplifyExpr', 'SimplifyInference'])
# 'LazyGradientInit', 'FirstOrderGradient', "EtaExpand", "PartialEvaluate", "ToMixedPrecision" ==> Error when compiling simple ResNet-18
# 'FakeQuantizationToInteger', 'ToANormalForm', 'ToBasicBlockNormalForm', 'ToGraphNormalForm'

gen_sequence_num = 500
gen_sequence_location = "./seq"