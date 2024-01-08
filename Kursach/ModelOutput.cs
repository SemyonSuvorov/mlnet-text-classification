using Microsoft.ML.Data;
using System.Numerics;

public class ModelOutput
{
    [ColumnName(@"Sentence")]
    public VBuffer<Single> Sentence { get;  set; }

    [ColumnName(@"Label")]
    public uint Label { get; set; }

    [ColumnName(@"PredictedLabel")]
    public float PredictedLabel { get; set; }

    [ColumnName(@"Score")]
    public float[] Score { get; set; }

}
