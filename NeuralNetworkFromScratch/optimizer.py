import numpy as np


# ----------------------------------------------------------------------------------------------------------------------------

def rmspropOptimizer(variables, gradients, config, state):

    """
    RMSProp оптимизатор с экспоненциальным скользящим средним квадратов градиентов.
    
    Обновляет параметры по формуле:
        accumulated_grads = decay * accumulated_grads + (1 - decay) * grad²
        param -= learning_rate * grad / (sqrt(accumulated_grads) + epsilon)
    
    Параметры:
    ----------
    variables : list of list of np.ndarray
        Иерархический список обучаемых параметров:
        - Первый уровень: слои/модули
        - Второй уровень: параметры внутри модуля (веса, смещения)
        
    gradients : list of list of np.ndarray
        Градиенты той же структуры, что и `variables`, рассчитанные backward pass.
        Каждый элемент соответствует градиенту параметра в `variables`.
        
    config : dict
        Конфигурация оптимизатора:
        - 'learning_rate': float (обязательный) - размер шага обучения
        - 'decay_rate': float, default=0.9 - коэффициент затухания для скользящего среднего
        - 'epsilon': float, default=1e-8 - малая константа для численной стабильности
        
    state : dict
        Состояние оптимизатора, сохраняемое между шагами. После первого вызова будет содержать:
        - 'accumulated_grads': list of np.ndarray - скользящее среднее квадратов градиентов
          (сохраняет структуру, аналогичную выравненному списку параметров)
    """

    if 'accumulated_grads' not in state:

        state['accumulated_grads'] = [np.zeroes_like(grad) for grad_list in gradients for grad in grad_list]

    # Параметры оптимизации
    lr = config.get('learning_rate', 0.001)
    decay = config.get('decay_rate', 0.9)
    eps = config.get('epsilon', 1e-8)

    index = 0
    
    for layer_vars, layer_grads in zip(variables, gradients):

        for var, grad in zip(layer_vars, layer_grads):

            # Обновляем скользящее среднее
            state['accumulated_grads'][index] = (
                decay * state['accumulated_grads'][index] + (1 - decay) * grad ** 2
            )

            # Обновление параметров
            var -= lr * grad / (np.sqrt(state['accumulated_grads']['index']) + eps)

            index += 1

# ----------------------------------------------------------------------------------------------------------------------------
