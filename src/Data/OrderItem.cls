/// 收费明细项，每一个就诊订单有零到多个收费明细项。
Class Data.OrderItem Extends %Persistent
{

/// 订单编号，关联Data.Order表的ID属性
Property OrderID As %String;

/// 收费项名称
Property ItemName As %String;

/// 收费项价格
Property Price As %Numeric;

/// 收费货币单位：CNY-人民币,USD-美元
Property Currency As %String(VALUELIST = ",CNY,USD");

Storage Default
{
<Data name="OrderItemDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>OrderID</Value>
</Value>
<Value name="3">
<Value>ItemName</Value>
</Value>
<Value name="4">
<Value>Price</Value>
</Value>
<Value name="5">
<Value>Currency</Value>
</Value>
</Data>
<DataLocation>^Data.OrderItemD</DataLocation>
<DefaultData>OrderItemDefaultData</DefaultData>
<IdLocation>^Data.OrderItemD</IdLocation>
<IndexLocation>^Data.OrderItemI</IndexLocation>
<StreamLocation>^Data.OrderItemS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
