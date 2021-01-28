pub mod models {

    use pyo3::{
        prelude::*,
        types::{IntoPyDict, PyList, PyTuple},
    };

    pub struct Model {
        pub name: String,
        model: PyObject,
        tokenizer: PyObject,
    }

    impl Model {
        pub fn new(name: String, location: String) -> Self {
            let gil = Python::acquire_gil();
            let py = gil.python();
            // Tensorflow
            let tf_models = py
                .import("tensorflow.keras.models")
                .expect("Tensorflow is not installed!");
            // Hub
            let hub = py
                .import("tensorflow_hub")
                .expect("Tensorflow hub is not installed!");

            let keras_layer: PyObject = hub.get("KerasLayer").unwrap().extract().unwrap();

            let tokenizer_module = PyModule::from_code(py, r#"
import tensorflow_hub as hub
import sys
sys.path.append('models')
from official.nlp.bert import tokenization

def load_tokenizer():
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", trainable=False)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    return tokenization.FullTokenizer(vocab_file, do_lower_case)
                "#, "tokenizer.py", "tokenizer").unwrap();

            let tokenizer = tokenizer_module
                .call0("load_tokenizer")
                .unwrap()
                .extract()
                .unwrap();

            // tf.keras.models.load_model('models/semantic_binary_classification2_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
            let kwargs = [(
                "custom_objects",
                [("KerasLayer", keras_layer)].into_py_dict(py),
            )]
            .into_py_dict(py);
            // load model
            let model = tf_models
                .call("load_model", PyTuple::new(py, &[location]), Some(kwargs))
                .unwrap()
                .extract()
                .unwrap();
            Self {
                name,
                model,
                tokenizer,
            }
        }

        pub fn check_plagiarism(
            &self,
            text_a: Vec<String>,
            text_b: Vec<String>,
        ) -> Result<Vec<(String, String)>, PyErr> {
            let gil = Python::acquire_gil();
            let py = gil.python();

            let preds_result = self.predict(text_a, text_b)?;

            let preds = preds_result.as_ref(py);
            let threshold = 0.7;

            let mut results = Vec::new();

            for pred in preds.iter() {
                let pred: f32 = pred.extract::<Vec<f32>>()?[0];
                let label = if pred >= threshold {
                    "Plagiarism"
                } else {
                    "Non"
                }
                .to_owned();
                let pred = format!("{:.2}", pred * 100.00);
                results.push((pred, label));
            }

            Ok(results)
        }

        fn predict(&self, text_a: Vec<String>, text_b: Vec<String>) -> PyResult<Py<PyList>> {
            Python::with_gil(|py| {
                let data_generator = PyModule::from_code(
                    py,
                    r#"
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')
from official.nlp.data import classifier_data_lib
import numpy as np
import re


label_list = [0, 1]
batch_size=32
max_seq_length = 128

def process_text(text):
    text = text if type(text) == str else text.decode("utf-8") 

    # put text in all lower case letters 
    text = text.lower()
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove html special characters (like &nbsp)
    text = re.sub(r'&\w+;\s*', '', text)
    # remove html tags
    text = re.sub(r'<[^>]*>', '', text)
    # remove digits
    text = re.sub(r'\d*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)

    # remove all non-alphanumeric chars
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # remove newlines/tabs, etc. so it's easier to match phrases, later
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub("  ", " ", text)
    text = re.sub("   ", " ", text)

    return text

# This provides a function to convert row to input features and label

def to_feature(text_a, text_b, label, label_list=label_list, max_seq_length=max_seq_length):
    example = classifier_data_lib.InputExample(guid=None, 
                                                text_a=process_text(text_a.numpy()), 
                                                text_b=process_text(text_b.numpy()), 
                                                label=label.numpy())
    global tokenizer
    feature = classifier_data_lib.convert_single_example(0, example, label_list, max_seq_length, tokenizer)

    return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

def to_feature_map(text_a, text_b, label):
    input_ids, input_mask, input_type_ids, label_id = tf.py_function(to_feature, inp=[text_a, text_b,  label], 
                                                                Tout=[tf.int32, tf.int32, tf.int32, tf.int32])
    input_ids.set_shape([max_seq_length])
    input_mask.set_shape([max_seq_length])
    input_type_ids.set_shape([max_seq_length])
    label_id.set_shape([])

    x = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }

    return (x, label_id)    

def data_generator(data, batch_size=batch_size, shuffle=False):
    with tf.device('/cpu:0'):

        data = data.map(
            tf.autograph.experimental.do_not_convert(
                func=to_feature_map
            ), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if shuffle:
            data = data.shuffle(512)

        return (data
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

def get_test_data(text_a, text_b, tokenizzzer):
    global tokenizer
    tokenizer = tokenizzzer
    # text_a and text_b are slices
    assert len(text_a) == len(text_b)
    test_data = tf.data.Dataset.from_tensor_slices((text_a, text_b, [0]*len(text_a)))
    return data_generator(test_data, batch_size=1)
                        "#,
                    "data_generator.py",
                    "data_generator",
                )?;

                let kwargs = [("tokenizzzer", &self.tokenizer)].into_py_dict(py);

                let test_data = data_generator.call(
                    "get_test_data",
                    PyTuple::new(py, &[text_a, text_b]),
                    Some(kwargs),
                )?;

                let preds: Py<PyAny>  = self
                    .model
                    .call_method1(py, "predict", PyTuple::new(py, &[test_data]))?
                    .extract(py)?;

                let preds: Py<PyList> = preds.call_method0(py, "tolist")?.extract(py)?;

                Ok(preds)
            })
        }
    }
}
