import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
import tensorflow as tf
from keras import layers, models

from simulatedata import generate_credit_risk_data
from sklearn.metrics import classification_report, roc_auc_score

# Use the function from before
df = generate_credit_risk_data(1000)

# Separate features and target
X = df.drop(columns='credit_risk')
y = df['credit_risk']

# Encode categorical feature
X['employment_status'] = LabelEncoder().fit_transform(X['employment_status'])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42,stratify=y)

def build_transformer_model(input_dim, num_classes=1):
    """This function implements a single transformer block for classification
    at a single point in time. Usually, transformers are used on time-series data.
    In this function, I use 20 heads in my multihead attention mechanism, this was
    an arbitrary choice, as was the size of the feed forward network portion of the
    transformer block. Other implementations for the transformer block are possible
    depending on skip connections."""

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)  # reshape for attention over features
    x = layers.MultiHeadAttention(num_heads=20,key_dim=input_dim, dropout=0.1)(x, x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(inputs+x)
    x = layers.Dropout(0.2)(x)
    logits = layers.Dense(1)(x)  # no activation here
    outputs = layers.Activation('sigmoid')(logits)
    # outputs = layers.Dense(1, activation='sigmoid')(x)  # binary classification

    model = models.Model(inputs=inputs, outputs=outputs)
    logit_model = models.Model(inputs=inputs, outputs=logits)
    adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=0.1)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'],)
    return model, logit_model
def build_logit_model(input_dim, num_classes=1):
    """This function implements a logistic regression in the keras framework."""
    inputs = layers.Input(shape=(input_dim,))
    logits = layers.Dense(1)(inputs)  # no activation here
    outputs = layers.Activation('sigmoid')(logits)
    # outputs = layers.Dense(1, activation='sigmoid')(inputs)  # binary classification

    model = models.Model(inputs=inputs, outputs=outputs)
    logit_model = models.Model(inputs=inputs, outputs=logits)
    adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=0.1)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'],)
    return model, logit_model


# Build the model
model, logit_model = build_transformer_model(input_dim=X_train.shape[1])
model2, logit_model2 = build_logit_model(input_dim=X_train.shape[1])
print(np.mean(y))
input('press enter to start training...')


# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1
)
history2 = model2.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
roc = roc_auc_score(y_test,model.predict(X_test))
logit_test_loss, logit_test_acc = model2.evaluate(X_test, y_test)
logit_roc = roc_auc_score(y_test,model2.predict(X_test))
print('==============================')
print(f"Transformer Test accuracy: {test_acc:.4f}")
print(f"Logit Test accuracy: {logit_test_acc:.4f}")
print(f"Transformer ROC AUC: {roc:.4f}")
print(f"logit ROC AUC: {logit_roc:.4f}")

print(classification_report(y_test, model.predict(X_test)>0.5))
print(classification_report(y_test, model2.predict(X_test)>0.5))
#print(model.predict(X_test))


# The nice thing about neural networks is that we can capture the gradient with respect to input
# directly from the model with the gradient tape class from tensorflow. This means that
# you can have a direct measurement of how the model is going to behave.

# You can see from this example that the model correctly recovers the direction and
# relative scale of the "true" coefficient vector.
def get_input_gradients(model, inputs):
    # Ensure inputs is a tensor with gradient tracking enabled
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.Variable(inputs)  # make it a variable to track gradients

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = model(inputs)

    # Compute gradient of output w.r.t. inputs
    grads = tape.gradient(outputs, inputs)
    return grads


# Example usage:
import numpy as np

x = X_test[0,:].reshape(-1,8)  # example input (batch size 1)
gradients = get_input_gradients(logit_model, x)
print('Transformer Gradient')
print(gradients.numpy())
gradients = get_input_gradients(logit_model2, x)
print('========================')
print('Logit Gradient')
print(gradients.numpy())