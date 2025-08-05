class AllReduce:
    def forward(self, ctx, input):
        pass

    def backward(self, ctx, grad_output):
        pass


class WarpMLS:
    def __init__(self, dst_h, dst_w, grid_size, pts, src, trans_ratio):
        self.dst_h = dst_h
        self.dst_w = dst_w
        self.grid_size = grid_size
        self.pts = pts
        self.src = src
        self.trans_ratio = trans_ratio

    def calc_delta(self):
        pass

    def gen_img(self):
        pass


class VQATokenPad:
    def __init__(self, infer_mode, max_seq_len, pad_to_max_seq_len, pad_token_label_id,
                 return_attention_mask, return_overflowing_tokens, return_special_tokens_mask,
                 return_token_type_ids, truncation_strategy):
        self.infer_mode = infer_mode
        self.max_seq_len = max_seq_len
        self.pad_to_max_seq_len = pad_to_max_seq_len
        self.pad_token_label_id = pad_token_label_id
        self.return_attention_mask = return_attention_mask
        self.return_overflowing_tokens = return_overflowing_tokens
        self.return_special_tokens_mask = return_special_tokens_mask
        self.return_token_type_ids = return_token_type_ids
        self.truncation_strategy = truncation_strategy


class VQATokenLabelEncode:
    def __init__(self, add_special_ids, contains_re, infer_mode, label2id, order_method, tokenizer, use_textline_bbox_info):
        self.add_special_ids = add_special_ids
        self.contains_re = contains_re
        self.infer_mode = infer_mode
        self.label2id = label2id
        self.order_method = order_method
        self.tokenizer = tokenizer
        self.use_textline_bbox_info = use_textline_bbox_info

    def filter_empty_contents(self, ocr_info):
        pass

    def split_bbox(self, bbox, text, tokenizer):
        pass

    def trans_poly_to_bbox(self, poly):
        pass
class UniMERNetLabelEncode:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def added_tokens_encoder(self, added_tokens_decoder):
        pass

    def encode(self, text, text_pair=None, return_token_type_ids=True,
               add_special_tokens=True, is_split_into_words=False):
        pass

    def set_truncation_and_padding(self, padding_strategy, truncation_strategy,
                                    max_length, stride, pad_to_multiple_of=None):
        pass


class UniMERNetImgDecode:
    def __init__(self, input_size, is_random_crop=False,
                 is_random_padding=False, is_random_resize=False):
        self.input_size = input_size
        self.is_random_crop = is_random_crop
        self.is_random_padding = is_random_padding
        self.is_random_resize = is_random_resize

    def crop_margin(self, img): pass
    def get_dimensions(self, img): pass
    def random_crop(self, img, crop_ratio): pass
    def random_resize(self, img): pass
    def resize(self, img, size): pass


class UniMERNetImageFormat:
    SPECIAL_TOKENS_ATTRIBUTES = []

    def __init__(self): pass


class UniMERNetDecode:
    def __init__(self, tokenizer, pad_token="",
                 pad_token_id=0, pad_token_type_id=0,
                 padding_side="right", pad_to_multiple_of=None,
                 special_tokens_map_extended=None):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.pad_token_type_id = pad_token_type_id
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        self.special_tokens_map_extended = special_tokens_map_extended

    def normalize_infer(self, s: str) -> str: pass
    def post_process(self, text: str) -> str: pass
    def remove_chinese_text_wrapping(self, formula): pass
    def token2str(self, token_ids) -> list: pass


class UniMERNetCollator:
    def __init__(self): pass


class TwoStepCosineDecay:
    def __init__(self, T_max1, T_max2, eta_min: float):
        self.T_max1 = T_max1
        self.T_max2 = T_max2
        self.eta_min = eta_min

    def get_lr(self):
        pass


class TwoStepCosine:
    def __init__(self, T_max1, T_max2, last_epoch: int, learning_rate, warmup_epoch):
        self.T_max1 = T_max1
        self.T_max2 = T_max2
        self.last_epoch = last_epoch
        self.learning_rate = learning_rate
        self.warmup_epoch = warmup_epoch


class TrainingStats:
    def __init__(self, window_size):
        self.window_size = window_size

    def get(self, extras=None): pass
    def update(self, stats): pass


class ToCHWImage:
    pass


class TextLineOrientationClassificationSubcommandExecutor:
    def __init__(self, subparser_name, wrapper_cls):
        self.subparser_name = subparser_name
        self.wrapper_cls = wrapper_cls


class TextLineOrientationClassification:
    default_model_name = "model"

    def get_cli_subcommand_executor(self):
        pass


class TextDetectionSubcommandExecutorMixin:
    pass


class TextDetectionMixin:
    pass


class TensorizeEntitiesRelations:
    def __init__(self, infer_mode: bool, max_seq_len: int):
        self.infer_mode = infer_mode
        self.max_seq_len = max_seq_len


class TelescopeLoss:
    def __init__(self, ce_loss, l1_loss, mse_loss, weight_table):
        self.ce_loss = ce_loss
        self.l1_loss = l1_loss
        self.mse_loss = mse_loss
        self.weight_table = weight_table

    def forward(self, pred, data):
        pass


class TableTree:
    def __init__(self, children: list, colspan=None, content=None, rowspan=None, tag=None):
        self.children = children
        self.colspan = colspan
        self.content = content
        self.rowspan = rowspan
        self.tag = tag

    def bracket(self):
        pass
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

    def get_metric(self): pass
    def reset(self): pass


class TableMatch:
    def __init__(self, filter_ocr_result: bool, use_master: bool):
        self.filter_ocr_result = filter_ocr_result
        self.use_master = use_master

    def get_pred_html(self, pred_structures, matched_index, ocr_contents): pass
    def get_pred_html_master(self, pred_structures, matched_index, ocr_contents): pass
    def match_result(self, dt_boxes, pred_bboxes): pass


class TableMasterLoss:
    def __init__(self, eps: float):
        self.eps = eps

    def forward(self, predicts, batch): pass


class TableClassificationSubcommandExecutor:
    def __init__(self, subparser_name, wrapper_cls):
        self.subparser_name = subparser_name
        self.wrapper_cls = wrapper_cls


class TableClassification:
    default_model_name = "table_model"

    def get_cli_subcommand_executor(self): pass


class TableCellsDetectionSubcommandExecutor:
    def __init__(self, subparser_name, wrapper_cls):
        self.subparser_name = subparser_name
        self.wrapper_cls = wrapper_cls


class TableCellsDetection:
    default_model_name = "table_cells_model"

    def get_cli_subcommand_executor(self): pass


class TableBoxEncode:
    def __init__(self, in_box_format: str, out_box_format: str):
        self.in_box_format = in_box_format
        self.out_box_format = out_box_format

    def xyxy2xywh(self, bboxes): pass
    def xyxyxyxy2xywh(self, boxes): pass


class TableAttentionLoss:
    def __init__(self, loc_weight: float, structure_weight: float):
        self.loc_weight = loc_weight
        self.structure_weight = structure_weight

    def forward(self, predicts, batch): pass


class TEDS:
    def __init__(self, n_jobs: int, structure_only: bool):
        self.n_jobs = n_jobs
        self.structure_only = structure_only

    def evaluate(self, pred, true): pass
    def batch_evaluate(self, pred_json, true_json): pass
    def batch_evaluate_html(self, pred_htmls, true_htmls): pass
    def load_html_tree(self, node, parent): pass
    def tokenize(self, node): pass


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
        # Aqui deve-se aplicar o uso de strokes conforme a língua
        pass


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

    def format_box(self, box): pass
    def get_metric(self): pass
    def prepare_bbox_metric_input(self, pred_label): pass
    def reset(self): pass
