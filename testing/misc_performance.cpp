#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

bool is_terminal(string history) {
	return history.back() == 'f';

}
int main() {
	srand(time(NULL));

	clock_t start = clock();
	vector<string> alphabet = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};

	for (int i = 0; i < 100000; i++)
	{
		string history = alphabet[rand() % alphabet.size()];
		is_terminal(history);
	}
	cout << "Time for procedural in s: " << (clock() - start) / (double)CLOCKS_PER_SEC << endl;
}