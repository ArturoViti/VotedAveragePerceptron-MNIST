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

                if pred == y[i]:  # Predizione corretta
                    c[k] += 1
                else:
                    v.append( np.add(v[k], np.dot(y[i], x[i])) )  # Sposta l'iperpiano
                    c.append(1)  # In caso di predizione errata, il peso viene messo a 1
                    k += 1

        self.V = v
        self.C = c
        self.k = k

    def predict( self, x ):
        """
            Effettua una predizione
            :param x: Record del test set
            :return: Restituisce la predizione (classificazione binaria)
        """
        s = 0
        for w, c in zip( self.V, self.C ):
            s = s + c * np.sign( np.dot(w, x) )
        return np.sign(s)

    def multiplePredict( self, X ):
        """
            :param X: Test Set
            :return: I segni delle predictions
        """
        s = np.zeros(len(X))
        for i in range( len(self.V) ):
            s += self.C[i] * np.sign( np.dot(self.V[i], X.T) ).T
        return np.sign(s)


class AveragePerceptron:
    """
        Modello Average Perceptron
    """
    def __init__( self, n_iter ):
        """
            Inizializza l'algoritmo
            :param n_iter: Numero di Epoche
        """
        self.n_iter = n_iter
        self.W = []

    def train( self, x, y ):
        """
            Addestra il modello
            :param x: Le istanze del Training Set
            :param y: Le label del Training Set
        """
        w = np.zeros_like(x[0])  # Inizializza il peso del modello
        c = 1  # Inizializza il contatore dei pesi
        w_sum = np.zeros_like(x[0])  # Inizializza la somma dei pesi per l'average perceptron
        c_sum = 0  # Inizializza il contatore per la somma dei pesi

        for epoch in range(self.n_iter):
            for i in range( len(x) ):
                pred = 1 if np.dot( w, x[i] ) > 0 else -1  # Stima del segno

                if pred == y[i]:  # Predizione corretta
                    c += 1
                else:
                    w = np.add( w, np.dot(y[i], x[i]) )
                    c = 1  # predizione errata => contatore dei pesi a 1

                # Aggiorna la somma dei pesi per l'average perceptron
                w_sum = np.add( w_sum, w )
                c_sum += c

        # Calcolo del peso medio
        w_avg = np.divide( w_sum, c_sum )
        self.W = w_avg  # Imposta il peso medio come peso del modello

    def predict( self, x ):
        """
            Effettua una predizione
            :param x: Record del test set
            :return: Restituisce la predizione (classificazione binaria)
        """
        prediction = np.sign( np.dot(self.W, x) )
        return int(prediction)

    def multiplePredict( self, X ):
        """
            Effettua predizioni su un Test Set
            :param X: Test Set
            :return: I segni delle predictions
        """
        predictions = []

        for instance in X:
            prediction = np.sign( np.dot(self.W, instance) )
            predictions.append(prediction)

        return predictions
