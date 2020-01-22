'use strict';

module.exports = (sequelize, DataTypes) => {
  const UserInput = sequelize.define('UserInput', {
      UserId: DataTypes.INTEGER,
      DWLimitLow: DataTypes.FLOAT,
      DWLimitHigh: DataTypes.FLOAT,
      BGLimit: DataTypes.FLOAT,
      WhiteSkedacityLimit: DataTypes.FLOAT,
      VIFLimit: DataTypes.FLOAT,
      ADFLimit: DataTypes.FLOAT,
      DynamicBacktestRange1: DataTypes.FLOAT,
      DynamicBacktestRange2: DataTypes.FLOAT,
      DynamicBacktestRange3: DataTypes.FLOAT,
      DynamicBacktestRange4: DataTypes.FLOAT,
      DynamicBacktestRange5: DataTypes.FLOAT,
      DynamicBacktestRange6: DataTypes.FLOAT,
      DynamicBacktestRange7: DataTypes.FLOAT,
      DynamicBacktestRange8: DataTypes.FLOAT,
      DynamicBacktestRange9: DataTypes.FLOAT,
      DynamicBacktestRange10: DataTypes.FLOAT,
      DynamicBacktest1Weight: DataTypes.FLOAT,
      DynamicBacktest2Weight: DataTypes.FLOAT,
      DynamicBacktest3Weight: DataTypes.FLOAT,
      DynamicBacktest4Weight: DataTypes.FLOAT,
      DynamicBacktest5Weight: DataTypes.FLOAT,
      DynamicBacktest6Weight: DataTypes.FLOAT,
      DynamicBacktest7Weight: DataTypes.FLOAT,
      DynamicBacktest8Weight: DataTypes.FLOAT,
      DynamicBacktest9Weight: DataTypes.FLOAT,
      DynamicBacktest10Weight: DataTypes.FLOAT,
      DynamicBacktestLongRange1: DataTypes.FLOAT,
      DynamicBacktestLongRange2: DataTypes.FLOAT,
      DynamicBacktestLongRange3: DataTypes.FLOAT,
      DynamicBacktestLongRange4: DataTypes.FLOAT,
      DynamicBacktestLongRange5: DataTypes.FLOAT,
      DynamicBacktestLongRange6: DataTypes.FLOAT,
      DynamicBacktestLongRange7: DataTypes.FLOAT,
      DynamicBacktestLongRange8: DataTypes.FLOAT,
      DynamicBacktestLongRange9: DataTypes.FLOAT,
      DynamicBacktestLongRange10: DataTypes.FLOAT
    }, {
      timestamps: false,
      freezeTableName: true,
      tableName: 'UserInput'
    });

      UserInput.associate = function(models) {
      models.UserInput.belongsTo(models.RunDetail, {
      onDelete: "CASCADE",
      foreignKey: "RunId"
    });
  };
  return UserInput;
};