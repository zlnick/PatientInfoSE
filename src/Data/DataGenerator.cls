Class Data.DataGenerator Extends %RegisteredObject
{

ClassMethod clearData() As %Status
{
    Set sc = $$$OK
    Set sc = ##Class(Data.OrderItem).%KillExtent()
    Set sc = ##Class(Data.Order).%KillExtent()
    Return sc
}

/// 按患者ID和就诊ID生成订单及明细
ClassMethod GenOrders(patientId As %String, entId As %String) As %Status
{
    Set sc = $$$OK
    // 2~5 Order per Encounter
    Set ordCount = $R(4)+2
    For i = 1:1:ordCount{
        #Dim order As Data.Order = ##Class(Data.Order).%New()
        Set order.Patient = patientId
        Set order.Encounter = entId
        Set ordTypeIdx = $R(11)+1
        if (0<ordTypeIdx<7){
            Set order.OrderCategory = "药品费"
        }else{
            if (ordTypeIdx<8){
                Set order.OrderCategory = "操作费"
            }else{
                Set order.OrderCategory = "耗材费"
            }
        }
        Set sc = order.%Save()
        // 1~5 items per Order
        Set itemCount = $R(5)+1
        for j = 1:1:itemCount{
            #Dim item As Data.OrderItem = ##Class(Data.OrderItem).%New()
            Set item.OrderID = order.%Id()
            Set item.Price = ##Class(%Library.PopulateUtils).Float(1)
            Set item.ItemName = ..Drug()
            Set item.Currency = ##Class(%Library.PopulateUtils).ValueList(",CNY")
            Set sc = item.%Save()
        }
        
    }
    Return sc
}

ClassMethod testRun() As %Status
{
    Set sc = ..clearData()
    Set patId = "Patient/794"
    Set entId = "Encounter/795"
    Set sc = ..GenOrders(patId,entId)
    Quit sc
}

/// Return a random product name (one of 4 possible values).
ClassMethod Drug() As %String
{
 s t1=$lb("对乙酰氨基酚","布洛芬","阿司匹林","阿莫西林","西替利嗪","氯雷他定","奥美拉唑","辛伐他汀","二甲双胍","赖诺普利","氯沙坦","沙丁胺醇","万托林","帕罗西汀","舍曲林","地西泮","阿普唑仑","可待因","曲马多","左甲状腺素","头孢拉定","罗红霉素","硝苯地平","硝酸甘油","氨溴索","氢氯噻嗪","格列美脲","氟康唑","阿奇霉素","多潘立酮")
 Quit $li(t1,$$$PRand($ll(t1))+1)
}

}
