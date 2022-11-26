from leduc.node import Node


def test_init():
    actions = ['F', 'C', 'R']
    node = Node(actions)

    strat = node.strategy()

    assert sum(strat.values()) == 1, node
    assert sum(node.strategy_sum.values()) == 1, node
    assert sum(node.regret_sum.values()) == 0, node


def test_weighting():
    actions = ['F', 'C', 'R']
    node = Node(actions)

    strat = node.strategy(.5)

    assert sum(strat.values()) == 1, node
    assert sum(node.strategy_sum.values()) == .5, node.strategy_sum
    assert sum(node.regret_sum.values()) == 0, node


def test_regrets():
    actions = ['F', 'C', 'R']
    node = Node(actions)

    node.regret_sum = {'F': .5, 'C': .5, 'R': 0}

    strat = node.strategy()

    assert strat == {'F': .5, 'C': .5, 'R': 0}, strat
    assert sum(node.strategy_sum.values()) == 1, node.strategy_sum


def test_average():
    actions = ['F', 'C', 'R']
    node = Node(actions)

    avg = node.avg_strategy()
    assert sum(avg.values()) == 1, avg

    strat = node.strategy(.5)

    assert sum(strat.values()) == 1, node
    assert sum(node.strategy_sum.values()) == .5, node.strategy_sum
    assert sum(node.regret_sum.values()) == 0, node

    avg = node.avg_strategy()

    assert sum(avg.values()) == 1, avg
