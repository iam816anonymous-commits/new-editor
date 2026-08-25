class PipelineError(Exception): pass
class ModelLoadError(PipelineError): pass
class FFmpegNotFoundError(PipelineError): pass
class RenderingError(PipelineError): pass
class TrajectoryError(PipelineError): pass
class QualityGateError(PipelineError): pass
class OutputError(PipelineError): pass
class OutputContractError(OutputError): pass
