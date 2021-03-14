pub mod models {

    use pyo3::{
        prelude::*,
        types::{PyList, PyTuple},
    };
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct PlagiarismResult {
        pub plagiarism: bool,
        pub accuracy: f32,
        pub text_a: String,
        pub text_b: String,
    }

    pub struct Model {
        pub name: String,
        model: PyObject,
        helpers: PyObject,
    }

    impl Model {
        pub fn new(name: String, location: String) -> Self {
            let gil = Python::acquire_gil();
            let py = gil.python();

            let helpers = PyModule::from_code(
                py,
                r#"
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
import re

class Helper:
    
    def __init__(self):
        self.label_list = [0, 1]
        self.batch_size = 32
        self.max_seq_length = 128
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3", trainable=False)
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


    def load_model(self, path, models_dir="saved_models/"):
        return tf.keras.models.load_model(models_dir+path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    def get_test_data(self, text_a, text_b):
        # text_a and text_b are slices
        assert len(text_a) == len(text_b)
        test_data = tf.data.Dataset.from_tensor_slices((text_a, text_b, [0]*len(text_a)))
        return self.data_generator(test_data, batch_size=1)

    def process_text(self, text):
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
        # don't remove them because the program is built for cyrillic symbols too
        # text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        # remove newlines/tabs, etc. so it's easier to match phrases, later
        text = re.sub(r"\t", " ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub("  ", " ", text)
        text = re.sub("   ", " ", text)

        return text

    # This provides a function to convert row to input features and label

    def to_feature(self, text_a, text_b, label):
        example = classifier_data_lib.InputExample(guid=None, 
                                                    text_a=self.process_text(text_a.numpy()), 
                                                    text_b=self.process_text(text_b.numpy()), 
                                                    label=label.numpy())
        global tokenizer
        feature = classifier_data_lib.convert_single_example(0, example, self.label_list, self.max_seq_length, self.tokenizer)

        return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

    def to_feature_map(self, text_a, text_b, label):
        input_ids, input_mask, input_type_ids, label_id = tf.py_function(self.to_feature, inp=[text_a, text_b,  label], 
                                                                    Tout=[tf.int32, tf.int32, tf.int32, tf.int32])
        input_ids.set_shape([self.max_seq_length])
        input_mask.set_shape([self.max_seq_length])
        input_type_ids.set_shape([self.max_seq_length])
        label_id.set_shape([])

        x = {
            'input_word_ids': input_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        }

        return (x, label_id)    

    def data_generator(self, data, batch_size=32, shuffle=False):
        with tf.device('/cpu:0'):

            data = data.map(
                tf.autograph.experimental.do_not_convert(
                    func=self.to_feature_map
                ), 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if shuffle:
                data = data.shuffle(512)

            return (data
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))


def init_helper():
    return Helper()
                "#,
                "helpers.py",
                "helpers",
            )
            .unwrap();

            // let helper: PyObject = helpers.get("Helper").unwrap().extract().unwrap();
            let helpers: PyObject = helpers.call0("init_helper").unwrap().extract().unwrap();

            // let helpers: PyObject = helper.call0(py).unwrap().extract(py).unwrap();

            let model = helpers
                .call_method1(py, "load_model", PyTuple::new(py, &[location]))
                .unwrap()
                .extract(py)
                .unwrap();

            Self {
                name,
                model,
                helpers,
            }
        }

        pub fn check_plagiarism(
            &self,
            text_a: Vec<String>,
            text_b: Vec<String>,
        ) -> Result<Vec<PlagiarismResult>, PyErr> {
            let gil = Python::acquire_gil();
            let py = gil.python();

            let preds_result = self.predict(text_a.clone(), text_b.clone())?;

            let preds = preds_result.as_ref(py);
            let threshold = 0.7;

            let mut results = Vec::new();

            for (i, pred) in preds.iter().enumerate() {
                let accuracy: f32 = pred.extract::<Vec<f32>>()?[0];
                let plagiarism = accuracy >= threshold;

                results.push(PlagiarismResult {
                    plagiarism,
                    accuracy,
                    text_a: text_a.get(i).unwrap().to_owned(),
                    text_b: text_b.get(i).unwrap().to_owned(),
                });
            }

            Ok(results)
        }

        fn predict(&self, text_a: Vec<String>, text_b: Vec<String>) -> PyResult<Py<PyList>> {
            Python::with_gil(|py| {
                let test_data = self.helpers.call_method1(
                    py,
                    "get_test_data",
                    PyTuple::new(py, &[text_a, text_b]),
                )?;

                let preds: Py<PyAny> = self
                    .model
                    .call_method1(py, "predict", PyTuple::new(py, &[test_data]))?
                    .extract(py)?;

                let preds: Py<PyList> = preds.call_method0(py, "tolist")?.extract(py)?;

                Ok(preds)
            })
        }
    }
}
