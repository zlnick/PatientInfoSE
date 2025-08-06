/// 就诊订单表，每次就诊过程会产生一到多条订单。
Class Data.Order Extends %Persistent
{

///  订单类型
Property OrderCategory As %String;

/// 患者ID，与FHIR资源的Key保持一致，如'Patient/123'
Property Patient As %String;

/// 就诊ID，与FHIR资源的Key保持一致，如'Encounter/456'
Property Encounter As %String;

Index idxCategory On OrderCategory [ Type = bitmap ];

Storage Default
{
<Data name="OrderDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>OrderCategory</Value>
</Value>
<Value name="3">
<Value>Patient</Value>
</Value>
<Value name="4">
<Value>Encounter</Value>
</Value>
</Data>
<DataLocation>^Data.OrderD</DataLocation>
<DefaultData>OrderDefaultData</DefaultData>
<IdLocation>^Data.OrderD</IdLocation>
<IndexLocation>^Data.OrderI</IndexLocation>
<StreamLocation>^Data.OrderS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
