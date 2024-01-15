import numpy as np


class VotedPerceptron:
    """
        Modello Voted Perceptron
    """
    def __init__( self, n_iter ):
        """
            Inizializza l'algoritmo
            :param n_iter: Numero di Epoche
        """
        self.n_iter = n_iter
        self.V = []
        self.C = []
        self.k = 0

    def train( self, x, y ):
        """
            Addestra il modello
            :param x: Le istanze del Training Set
            :param y: Le label del Training Set
        """
        k = 0
        # Inizializza il vettore delle predizioni a 0 tante quante sono le colonne di un'istanza del Training Set
        v = [np.zeros_like(x)[0]]
        c = [0]

        for epoch in range( self.n_iter ):
            for i in range( len(x) ):
                pred = 1 if np.dot(v[k], x[i]) > 0 else -1  # Stima del segno

                if pred == y[i]:  # Se la predizione è corretta, incrementa il peso
                    c[k] += 1
                else:
                    v.append( np.add(v[k], np.dot(y[i], x[i])) )  # Sposta l'iperpiano
                    c.append(1)  # In caso di predizione errata, il peso viene messo a 1
                    k += 1

        self.V = v
        self.C = c
        self.k = k

    def predict( self, x, avgPerceptronModel=False ):
        """
            Effettua una predizione
            :param x: Record del test set
            :param avgPerceptronModel: Se eseguire una predizione Voted o nella variante Average
            :return: Restituisce la predizione (classificazione binaria)
        """
        s = 0
        if avgPerceptronModel is False:
            for w, c in zip( self.V, self.C ):
                s = s + c * np.sign( np.dot(w, x) )
        else:
            for w, c in zip( self.V, self.C ):
                s = s + c * np.dot(w, x)
        return np.sign(s)

    def multiplePredict( self, X, avgPerceptronModel=False ):
        """
            :param X: Test Set
            :param avgPerceptronModel: Indica se fare predizioni con il modello Voted o Average
            :return: I segni delle predictions
        """
        s = np.zeros(len(X))

        if avgPerceptronModel is False:
            for i in range( len(self.V) ):
                s += self.C[i] * np.sign( np.dot(self.V[i], X.T) ).T
        else:
            for i in range( len(self.V) ):
                s += self.C[i] * np.dot( self.V[i], X.T ).T
        return np.sign(s)
