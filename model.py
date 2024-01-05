import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('ruouvang.csv')
#xử lí dữ liệu
#xử lí dữ liệu rỗng
# thay thế dữ liệu rỗng bàng giá trị trung bình
# Lọc các cột có kiểu dữ liệu số
numeric_columns = data.select_dtypes(include=[np.number])

# Điền giá trị rỗng trong các cột số bằng giá trị trung bình của cột đó
data[numeric_columns.columns] = data[numeric_columns.columns].fillna(data[numeric_columns.columns].mean())
data.isnull().sum()
#chuyển thành nhị phân
data['type'].replace(to_replace=['white','red'], value=[0,1],inplace=True)
data['type'].value_counts()

# cân bằng dữ liệu
from sklearn.utils import resample

# Tách dữ liệu thành các lớp riêng biệt
white_wines = data[data['type'] == 0]
red_wines = data[data['type'] == 1]

# Tăng mẫu lớp thiểu số (ví dụ: red wines) để cân bằng tỷ lệ với lớp đa số
red_wines_upsampled = resample(red_wines, replace=True, n_samples=len(white_wines), random_state=42)

# Kết hợp dữ liệu tăng mẫu với dữ liệu gốc của lớp đa số (white wines)
balanced_data = pd.concat([white_wines, red_wines_upsampled])

# Kiểm tra số lượng mẫu trong từng lớp
print(balanced_data['type'].value_counts())
# lưu data đã được cân bằng
balanced_data.to_csv('balanced_data.csv', index=False)

# Xác định features (đặc trưng) và target (mục tiêu)
X = balanced_data.drop('quality', axis=1)  # features
y = balanced_data['quality']  # target

from sklearn.model_selection import train_test_split

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% dữ liệu cho huấn luyện, 20% cho kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Khởi tạo và huấn luyện mô hình RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
rf_pred = rf.predict(X_test)
print("Dự đoán từ RandomForestClassifier: ", rf_pred)

# lưu Model
import pickle
# Lưu mô hình Random Forest Classifier
pickle.dump(rf, open('random_forest_model.pkl', 'wb'))