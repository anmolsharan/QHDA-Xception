
# import tensorflow as tf
# import pennylane as qml
# from tensorflow.keras.layers import Layer

# class QuantumLayer(Layer):
#     def __init__(self, n_qubits: int = 4, **kwargs):
#         super().__init__(**kwargs)
#         self.n_qubits = n_qubits
#         self.dev = qml.device("default.qubit", wires=n_qubits)

#         @qml.qnode(self.dev, interface="tf", diff_method="backprop")
#         def _circuit(features, weights):
#             qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
#             for i in range(n_qubits - 1):
#                 qml.CNOT(wires=[i, i + 1])
#             for i in range(n_qubits):
#                 qml.RY(weights[i], wires=i)
#             return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

#         self._circuit = _circuit

#     def build(self, input_shape):
#         self.q_weights = self.add_weight(
#             shape=(self.n_qubits,),
#             initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
#             trainable=True,
#             dtype=tf.float32,
#             name="q_weights",
#         )
#         super().build(input_shape)

#     def call(self, inputs):
#         inputs = tf.cast(inputs, tf.float32)
#         def _wrapper(x):
#             out = self._circuit(x, self.q_weights)
#             return tf.cast(tf.stack(out), tf.float32)
#         return tf.map_fn(
#             _wrapper,
#             inputs,
#             fn_output_signature=tf.TensorSpec(shape=(self.n_qubits,), dtype=tf.float32),
#         )

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.n_qubits)

#     def get_config(self):
#         cfg = super().get_config()
#         cfg["n_qubits"] = self.n_qubits
#         return cfg


import tensorflow as tf
import pennylane as qml
from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable()
class QuantumLayer(Layer):

    def __init__(self, n_qubits=4, **kwargs):
        super().__init__(**kwargs)

        self.n_qubits = n_qubits

        self.dev = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        @qml.qnode(
            self.dev,
            interface="tf",
            diff_method="backprop"
        )
        def circuit(features, weights):

            qml.AngleEmbedding(
                features,
                wires=range(n_qubits),
                rotation="Y"
            )

            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            for i in range(n_qubits):
                qml.RY(weights[i], wires=i)

            return [
                qml.expval(qml.PauliZ(i))
                for i in range(n_qubits)
            ]

        self.circuit = circuit

    def build(self, input_shape):

        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(self.n_qubits,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.1,
                maxval=0.1
            ),
            trainable=True,
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, inputs):

        inputs = tf.cast(
            inputs,
            tf.float32
        )

        def process(x):

            result = self.circuit(
                x,
                self.q_weights
            )

            return tf.cast(
                tf.stack(result),
                tf.float32
            )

        return tf.map_fn(
            process,
            inputs,
            fn_output_signature=tf.TensorSpec(
                shape=(self.n_qubits,),
                dtype=tf.float32
            )
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "n_qubits": self.n_qubits
        })

        return config
