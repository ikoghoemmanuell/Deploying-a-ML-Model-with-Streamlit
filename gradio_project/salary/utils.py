# Source: https://github.com/eaedk/salary_prediction/blob/master/code/utils.py


def apply_processing(
    dataframe,
    cateogrical_imputer,
    numerical_imputer,
    encoder,
    scaler,
    numerical_cols=["name_numerical_col_001"],
    categorical_cols=["name_categorical_col_001"],
):
    "Straightforward pipeline to apply the preprocessing and the feature engineering over and over again"

    df_ = dataframe.copy()

    df_[categorical_cols] = cateogrical_imputer.transform(df_[categorical_cols])
    df_[numerical_cols] = numerical_imputer.transform(df_[numerical_cols])

    encoded_cols = list(encoder.get_feature_names_out())
    df_[encoded_cols] = encoder.transform(df_[categorical_cols])

    useful_cols = numerical_cols + encoded_cols
    df_[useful_cols] = scaler.transform(df_[useful_cols])

    return df_[useful_cols]
