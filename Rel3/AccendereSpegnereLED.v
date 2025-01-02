module LEDcomand (
    input CLK100MHZ,
    input btnC,
    output reg[15:0] LED
);
reg obtnC, onoff;
initial begin
    LED = 16'b0000_0000_0000_0001;
    obtnC <= btnC;
    onoff = 0;
end

always @(posedge CLK100MHZ) begin
    if(!btnC & obtnC) begin
        LED[0] = onoff;
    end
end
endmodule