class TableSystem:
    def __init__(self, args, config, input_tensor, match,
                 predictor, table_structurer,
                 text_detector=None, text_recognizer=None):
        self.args = args
        self.config = config
        self.input_tensor = input_tensor
        self.match = match
        self.predictor = predictor
        self.table_structurer = table_structurer
        self.text_detector = text_detector
        self.text_recognizer = text_recognizer


class TableStructurer:
    def __init__(self, args, autolog, config, input_tensor,
                 postprocess_op, predictor, preprocess_op, use_onnx):
        self.args = args
        self.autolog = autolog
        self.config = config
        self.input_tensor = input_tensor
        self.postprocess_op = postprocess_op
        self.predictor = predictor
        self.preprocess_op = preprocess_op
        self.use_onnx = use_onnx


class TableStructureMetric:
    def __init__(self):
        self.all_num = 0
        self.anys_dict = {}
        self.correct_num = 0
        self.main_indicator = ""
        self.token_nums = 0
        self.del_thead_tbody = False

    def get_metric(self):
        return {
            "main_indicator": self.main_indicator,
            "correct": self.correct_num,
            "total": self.all_num,
            "tokens": self.token_nums
        }

    def reset(self):
        self.all_num = 0
        self.anys_dict.clear()
        self.correct_num = 0
        self.token_nums = 0


class TableMatch:
    def __init__(self, filter_ocr_result: bool, use_master: bool):
        self.filter_ocr_result = filter_ocr_result
        self.use_master = use_master

    def get_pred_html(self, pred_structures, matched_index, ocr_contents):
        raise NotImplementedError

    def get_pred_html_master(self, pred_structures, matched_index, ocr_contents):
        raise NotImplementedError

    def match_result(self, dt_boxes, pred_bboxes):
        raise NotImplementedError


class TableMasterLoss:
    def __init__(self, eps: float):
        self.eps = eps

    def forward(self, predicts, batch):
        raise NotImplementedError


class TableClassificationSubcommandExecutor:
    def __init__(self, subparser_name, wrapper_cls):
        self.subparser_name = subparser_name
        self.wrapper_cls = wrapper_cls


class TableClassification:
    @classmethod
    def default_model_name(cls):
        return "table_model"

    def get_cli_subcommand_executor(self):
        raise NotImplementedError


class TableCellsDetectionSubcommandExecutor:
    def __init__(self, subparser_name, wrapper_cls):
        self.subparser_name = subparser_name
        self.wrapper_cls = wrapper_cls


class TableCellsDetection:
    @classmethod
    def default_model_name(cls):
        return "table_cells_model"

    def get_cli_subcommand_executor(self):
        raise NotImplementedError


class TableBoxEncode:
    def __init__(self, in_box_format: str, out_box_format: str):
        self.in_box_format = in_box_format
        self.out_box_format = out_box_format

    def xyxy2xywh(self, bboxes):
        return [self._convert_box(box, "xyxy2xywh") for box in bboxes]

    def xyxyxyxy2xywh(self, boxes):
        return [self._convert_box(box, "xyxyxyxy2xywh") for box in boxes]

    def _convert_box(self, box, method):
        # placeholder conversion logic
        return box


class TableAttentionLoss:
    def __init__(self, loc_weight: float, structure_weight: float):
        self.loc_weight = loc_weight
        self.structure_weight = structure_weight

    def forward(self, predicts, batch):
        raise NotImplementedError


class TEDS:
    def __init__(self, n_jobs: int, structure_only: bool):
        self.n_jobs = n_jobs
        self.structure_only = structure_only

    def evaluate(self, pred, true):
        raise NotImplementedError

    def batch_evaluate(self, pred_json, true_json):
        raise NotImplementedError

    def batch_evaluate_html(self, pred_htmls, true_htmls):
        raise NotImplementedError

    def load_html_tree(self, node, parent):
        raise NotImplementedError

    def tokenize(self, node):
        raise NotImplementedError


class StructureSystem:
    def __init__(self, mode, formula_system=None, image_orientation_predictor=None,
                 kie_predictor=None, layout_predictor=None, recovery=None,
                 return_word_box=None, table_system=None, text_system=None):
        self.mode = mode
        self.formula_system = formula_system
        self.image_orientation_predictor = image_orientation_predictor
        self.kie_predictor = kie_predictor
        self.layout_predictor = layout_predictor
        self.recovery = recovery
        self.return_word_box = return_word_box
        self.table_system = table_system
        self.text_system = text_system


class StrokeFocusLoss:
    def __init__(self, ce_loss, mse_loss, dic: dict,
                 stroke_dicts: dict, stroke_alphabets: dict,
                 language: str = "english"):
        self.ce_loss = ce_loss
        self.mse_loss = mse_loss
        self.dic = dic
        self.stroke_dicts = stroke_dicts
        self.stroke_alphabets = stroke_alphabets
        self.language = language.lower()

        if self.language not in self.stroke_dicts:
            raise ValueError(f"Idioma não suportado: {self.language}")

        self.active_stroke_dict = self.stroke_dicts[self.language]
        self.active_stroke_alphabet = self.stroke_alphabets[self.language]

    def forward(self, pred, data):
        stroke_penalty = sum(len(self.active_stroke_dict.get(char, [])) for char in data)
        return self.ce_loss + self.mse_loss + stroke_penalty


class Step:
    def __init__(self, gamma, last_epoch: int, learning_rate, step_size, warmup_epoch):
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.warmup_epoch = warmup_epoch


class TableMetric:
    def __init__(self, bbox_metric=None, box_format: str = "xywh",
                 main_indicator: str = "", structure_metric=None):
        self.bbox_metric = bbox_metric
        self.box_format = box_format
        self.main_indicator = main_indicator
        self.structure_metric = structure_metric

    def format_box(self, box):
        return {
            "box": box,
            "format": self.box_format
        }

    def get_metric(self):
        return {
            "indicator": self.main_indicator,
            "bbox_metric": self.bbox_metric,
            "structure": self.structure_metric
        }

    def prepare_bbox_metric_input(self, pred_label):
        raise NotImplementedError

    def reset(self):
        self.bbox_metric = None
        self.structure_metric = None
