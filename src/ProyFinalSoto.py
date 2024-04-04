"""
Este script entrena un modelo de clasificación de dígitos escritos a mano utilizando el conjunto de datos de dígitos de sklearn.
El modelo se entrena utilizando una búsqueda de hiperparámetros y validación cruzada.
Finalmente, se visualizan las predicciones del modelo en un subconjunto de las imágenes de prueba.....
"""
# Librerías utilizadas:
import sklearn.datasets as datasets # Conjuntos de datos integrados y utilidades para cargar datos.
from sklearn.model_selection import train_test_split, GridSearchCV # Funciones para dividir conjuntos de datos y validación cruzada.
import sklearn.svm as svm # Implementación de Support Vector Machines (SVM) para clasificación.
from sklearn.metrics import accuracy_score, classification_report # Métricas para evaluar el rendimiento de modelos.
import matplotlib.pyplot as plt # Creación de visualizaciones gráficas.

def cargar_datos():
    """
    Cargar el conjunto de datos de dígitos utilizando sklearn.datasets.load_digits().
    
    Returns:
        digits: El conjunto de datos de dígitos.
    """
    digits = datasets.load_digits()
    return digits

def entrenar_modelo(X_train, y_train):
    """
    Entrenar el clasificador SVM utilizando validación cruzada y búsqueda de hiperparámetros.
    
    Args:
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        
    Returns:
        clf: El modelo SVM entrenado.
    """
    parametros = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
    clf = GridSearchCV(svm.SVC(random_state=42), parametros, cv=5)
    clf.fit(X_train, y_train)
    return clf

def evaluar_modelo(clf, X_test, y_test):
    """
    Evaluar el modelo entrenado en el conjunto de prueba.
    
    Args:
        clf: El modelo entrenado.
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        
    Returns:
        precision: Precisión del modelo.
        reporte: Informe de clasificación.
    """
    y_pred = clf.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)
    return precision, reporte

def visualizar_resultados(clf, digits, num_imagenes=10):
    """
    Visualizar imágenes de dígitos con sus predicciones.
    
    Args:
        clf: El modelo entrenado.
        digits: El conjunto de datos de dígitos.
        num_imagenes: Número de imágenes para visualizar (por defecto 10).
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6), constrained_layout=True)
    fig.patch.set_facecolor('lightgray')
    fig.suptitle("Predicciones de Dígitos Escritos a Mano", fontsize=20, color='navy')
    for ax, (imagen, etiqueta) in zip(axes.ravel(), zip(digits.images[:num_imagenes], digits.target[:num_imagenes])):
        prediccion = clf.predict(imagen.reshape(1, -1))[0]
        ax.imshow(imagen, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Real: {etiqueta}\nPredicción: {prediccion}', fontsize=12, color='darkorange')
        ax.axis('off')
    plt.show()

def main():
    """
    Función principal que carga los datos, entrena el modelo, evalúa su precisión y visualiza los resultados.
    """
    # Cargar los datos
    digits = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    clf = entrenar_modelo(X_train, y_train)
    
    # Evaluar el modelo
    precision, reporte = evaluar_modelo(clf, X_test, y_test)
    print("\nPrecisión del Modelo:", precision)
    print("\nInforme de Clasificación:\n", reporte)
    
    # Visualizar los resultados
    visualizar_resultados(clf, digits)

if __name__ == "__main__":
    main()
