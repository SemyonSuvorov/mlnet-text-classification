using Microsoft.ML;
using Microsoft.ML.Data;

// Initialize MLContext
MLContext mlContext = new()
{
    GpuDeviceId = 0,
    FallbackToCpu = true
};

// Load the data source
Console.WriteLine("Loading data...");

/** MODEL TRAINING ****************************************************************************/

// To evaluate the effectiveness of machine learning models we split them into a training set for fitting
// and a testing set to evaluate that trained model against unknown data

IDataView trainData = mlContext.Data.LoadFromTextFile<ModelInput>(
    "train.tsv",
    separatorChar: '\t',
    hasHeader: false
);
IDataView testData = mlContext.Data.LoadFromTextFile<ModelInput>(
    "test.tsv",
    separatorChar: '\t',
    hasHeader: false
);

// Create a pipeline for training the model
//
/*
var pipeline = mlContext.Transforms.Concatenate("Feature", "Sentence", "Description")
                    .Append(mlContext.Transforms.Text.NormalizeText("Feature"))
                    .Append(mlContext.Transforms.Text.TokenizeIntoWords("Feature"))
                    .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Feature"))
                    .Append(mlContext.Transforms.Conversion.MapValueToKey("Feature"))
                    .Append(mlContext.Transforms.Text.ProduceNgrams("Feature"))
                    .Append(mlContext.Transforms.NormalizeLpNorm("Feature"))
                    .Append(mlContext.Clustering.Trainers.KMeans("Feature", numberOfClusters:4));
*/
/*
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                          .Append(mlContext.Transforms.Concatenate("Feature", "Sentence", "Description"))
                          .Append(mlContext.Transforms.Text.FeaturizeText("Feature", new TextFeaturizingEstimator.Options
                          {
                              KeepDiacritics = true,
                              CaseMode = TextNormalizingEstimator.CaseMode.Lower,
                          }, "Feature"))
                          .Append(mlContext.Clustering.Trainers.KMeans("Feature", numberOfClusters: 4));

*/

var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                          .Append(mlContext.Transforms.Concatenate("Feature", "Sentence", "Description"))
                          .Append(mlContext.Transforms.Text.TokenizeIntoWords("Feature"))
                          .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Feature"))
                          .Append(mlContext.Transforms.Conversion.MapValueToKey("Feature"))
                          .Append(mlContext.Transforms.Text.ProduceNgrams("Feature"))
                          .Append(mlContext.Transforms.NormalizeLpNorm("Feature"))
                          .Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(featureColumnName: "Feature"))
                          .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

Console.WriteLine("Training model...");
ITransformer model = pipeline.Fit(trainData);
Console.WriteLine("Evaluating model performance...");

// We need to apply the same transformations to our test set so it can be evaluated via the resulting model
IDataView transformedTest = model.Transform(testData);
MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

// Display Metrics
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");
Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");
Console.WriteLine($"Log Loss: {metrics.LogLoss}");
Console.WriteLine();

Console.WriteLine("Classes:");
foreach (ArticleIntents value in Enum.GetValues<ArticleIntents>())
{
    Console.WriteLine($"{((int)value)}: {value}");
}
Console.WriteLine();

Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
/** PREDICTION GENERATION *********************************************************************/
// Generate a prediction engine
Console.WriteLine("Creating prediction engine...");

PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Ready to generate predictions.");

// Generate a series of predictions based on user input
string input;
do
{
    Console.WriteLine();
    Console.WriteLine("What do you want to say ? (Type Q to Quit)");
    input = Console.ReadLine()!;

    // Get a prediction
    ModelInput sampleData = new(input);
    ModelOutput result = engine.Predict(sampleData);
    // Print classification
    float maxScore = result.Score[(uint)result.PredictedLabel];
    Console.WriteLine($"Matched intent {(ArticleIntents)result.PredictedLabel} with score of {maxScore:f2}");
    Console.WriteLine();
}
while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");

