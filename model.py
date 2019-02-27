from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from feature_extraction import makeFeatures, getTfidfVectorizer


if __name__ == "__main__":
    data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label', 'body_text'], header=None)
    data = makeFeatures(data)
    # vectorizer, X_vector = getTfidfVectorizer(data)
    # X_tfidf_features = pd.concat([data['body_text'], data['punctuation_percentage'], pd.DataFrame(X_vector.toarray())], axis=1)

    # rf = RandomForestClassifier()
    # param = {'n_estimators': [10, 150, 300],
    #      'max_depth': [30, 60, 90, None]}

    # gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)
    # gs_fit = gs.fit(X_vector, data['label'])

    X=data[['body_text', 'body_len', 'punctuation_percentage', 'capital_words', 'link_count']]
    y=data['label']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    X_vectorizer = getTfidfVectorizer(X_train)
    X_train_transformed = X_vectorizer.transform(X_train['body_text'])
    X_test_transformed = X_vectorizer.transform(X_test['body_text'])

    X_train_vector = pd.concat([X_train[['body_len', 'punctuation_percentage', 'capital_words', 'link_count']].reset_index(drop=True), 
           pd.DataFrame(X_train_transformed.toarray())], axis=1)
    X_test_vector = pd.concat([X_test[['body_len', 'punctuation_percentage', 'capital_words', 'link_count']].reset_index(drop=True), 
           pd.DataFrame(X_test_transformed.toarray())], axis=1)

    rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

    rf_model = rf.fit(X_train_vector, y_train)
    y_pred = rf_model.predict(X_test_vector)

    precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')
    print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_test,y_pred), 3)))


    cm = confusion_matrix(y_test, y_pred)
    class_label = ["ham", "spam"]
    df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()