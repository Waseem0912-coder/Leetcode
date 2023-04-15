#include <stdio.h>
#include <vector>
#include <algorithm>
using namespace std;

int main(){
    vector<int> prices = {7,1,5,3,6,4};
    int min = prices[0];
    int profit = 0;
    for(int i=1;i<prices.size();i++){

    //checks and upadtes the profit if price less than min

        if(prices[i]>min){
           profit = max(profit,prices[i]-min);
        }
        else if (prices[i]<min){
            min = prices[i];
        }
    }
    printf("%d",profit);
    return 0;
}
