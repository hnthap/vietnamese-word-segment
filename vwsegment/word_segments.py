from typing import Literal

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)


class WordSegments(object):

    def __init__(
        self,
        texts: str | list[str],
        lang: Literal['vi']='vi',
        *,
        case=False,
        device=None,
        truncate=True,
        max_length=512,
        torch_dtype='float16',
        batch_size=128
    ):
        '''
        Initialize ViWordSegments object with given texts and optional
        parameters.

        Params:
            texts (str or list[str]):
                The input text(s) to be segmented.
            lang (Literal['vi'], optional):
                Language of the text(s). Default is 'vi'.
            case (bool, optional):
                If True, preserve the original case of words; lowercase all,
                otherwise. Default is False.
            device (str, optional):
                The device to run the model on. If None, use the default
                device.
            truncate (bool, optional):
                If True, truncate the input to max_length. Default is True.
            max_length (int, optional):
                The maximum length of the input text. Default is 512.
            torch_dtype (str, optional):
                The data type for the model's weights. Default is 'float16'.
            batch_size (int, optional):
                The batch size for processing texts. Default is 128.

        Returns:
            None
        '''
        self.lang = lang
        self._load_model(lang, device=device, truncate=truncate,
                         max_length=max_length, torch_dtype=torch_dtype)
        if isinstance(texts, str):
            texts = [texts]
        self.texts = texts
        self._segmented = self._segment(texts, batch_size=batch_size)
        if case:
            self._norm_segmented = self._segmented
        else:
            self._norm_segmented = [
                list(map(lambda s: s.lower(), segments))
                for segments in self._segmented
            ]


    def __getitem__(self, index):
        return self._norm_segmented[index]

    
    def _segment(self, texts: list[str], *, batch_size) -> list[list[str]]:
        results = self._pipe(texts, batch_size=batch_size)
        segments = []
        for doc in results:
            tokens = ''
            for e in doc:
                word = e['word']
                if '##' in word:
                    tokens += word.replace('##', '')
                elif e['entity'] == 'I':
                    tokens += '_' + word
                else:
                    tokens += ' ' + word
            segments.append(tokens.strip().split())
        return segments

    
    @classmethod
    def _load_model(cls, lang, *, device, truncate, max_length, torch_dtype):
        
        assert lang == 'vi', \
            'Only Vietnamese is supported. lang can only be "vi".'
        
        reloads_tokenizer = cls._needs_reload_tokenizer(truncate, max_length)
        reloads_model = cls._needs_reload_model(torch_dtype, max_length)
        reloads_pipe = cls._needs_reload_pipe(device)

        if reloads_tokenizer:
            cls._tokenizer = AutoTokenizer.from_pretrained(
                'NlpHUST/vi-word-segmentation',
                truncate=truncate,
                model_max_length=max_length,
            )
            cls.truncate = truncate
            cls.max_length = max_length

        if reloads_model:
            cls._model = AutoModelForTokenClassification.from_pretrained(
                'NlpHUST/vi-word-segmentation',
                torch_dtype=torch_dtype,
                max_length=max_length,
            )
            cls.torch_dtype = torch_dtype
            cls.max_length = max_length

        if reloads_pipe or reloads_tokenizer or reloads_model:
            cls._pipe = pipeline(
                'token-classification',
                model=cls._model,
                tokenizer=cls._tokenizer,
                device=device,
            )
            cls.device = device


    @classmethod
    def _needs_reload_tokenizer(cls, truncate, max_length):
        return (
            not hasattr(cls, '_tokenizer')
            or not hasattr(cls, 'truncate')
            or cls.truncate != truncate
        )


    @classmethod
    def _needs_reload_model(cls, torch_dtype, max_length):
        return (
            not hasattr(cls, '_model')
            or not hasattr(cls, 'torch_dtype')
            or cls.torch_dtype != torch_dtype
            or not hasattr(cls, 'max_length')
            or cls.max_length != max_length
        )

    
    @classmethod
    def _needs_reload_pipe(cls, device):
        return (
            not hasattr(cls, '_pipe')
            or not hasattr(cls, 'device')
            or cls.device != device
        )

