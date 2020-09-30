{
  "dataset_reader": {
    "type": "ner",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": std.extVar("PROJECT_DATA") + "/bert_pretrained_models/YOUR_BERT_MODEL",
          "do_lowercase": false,
          "use_starting_offsets": true,
          "truncate_long_sequences": false,
      },
    }
  },
  "train_data_path": std.extVar("PROJECT_DATA") + "/datasets/ner/TRAINING_LANGUAGE_A",
  "validation_data_path": std.extVar("PROJECT_DATA") + "/datasets/ner/DEV_LANGUAGE_A",
  "test_data_path": std.extVar("PROJECT_DATA") + "/datasets/ner/TEST_LANGUAGE_B",
  "evaluate_on_test": true,
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
        },
        "token_embedders": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": std.extVar("PROJECT_DATA") + "/bert_pretrained_models/YOUR_BERT_MODEL",
            },
        }
    },
    "encoder": {
        "type": "lstm",
        "input_size": 768,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32,
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}