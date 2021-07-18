using CoffeeRecommender.Domain.Services;
using Microsoft.ML;

namespace CoffeeRecommender.Application
{
    class Program
    {
        static void Main(string[] args)
        {
            var mLContext = new MLContext();
            var mlService = new MLService();
            IDataView trainingDataView = mlService.LoadTrainingData(mLContext);
            IDataView testDataView = mlService.LoadTestData(mLContext);
            ITransformer model = mlService.BuildAndTrainModel(mLContext, trainingDataView);
            mlService.EvaluateModel(mLContext, testDataView, model);
        }
    }
}